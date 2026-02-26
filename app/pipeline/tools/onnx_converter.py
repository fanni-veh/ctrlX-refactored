import time
import numpy as np
import pandas as pd
from skl2onnx import convert_sklearn
import onnxruntime
from skl2onnx import update_registered_converter
from catboost import CatBoostClassifier
from onnx.helper import get_attribute_value
from skl2onnx._parse import _apply_zipmap, _get_sklearn_operator_name
from catboost.utils import convert_to_onnx_object

from skl2onnx.common.data_types import (
    FloatTensorType,
    Int64TensorType,
    guess_tensor_type,
)

from skl2onnx.common.shape_calculator import (
    calculate_linear_classifier_output_shapes,
)

from app.pipeline.tools.hdbscan_clustering import HDBSCAN_Clustering


def sk_to_onnx(input_dim, model):
    start = time.perf_counter()
    initial_type = [('input', FloatTensorType([None, input_dim]))]
    if isinstance(model, HDBSCAN_Clustering):
        # Use custom converter for HDBSCAN
        onnx_model = model.to_onnx().SerializeToString()
    else:
        onnx_model = convert_sklearn(model, initial_types=initial_type, target_opset={'': 15, 'ai.onnx.ml': 3}).SerializeToString()
    end = time.perf_counter()
    print(f"Time taken for sk_to_onnx: {end-start:2f} s")
    return onnx_model


def onnx_predict(onnx_model, X_valid, cycle_ids):

    # Separate values and cycle_id
    X_valid_array = X_valid.values.astype(np.float32)

    # ONNX prediction
    session = onnxruntime.InferenceSession(onnx_model)
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    pred_onnx = session.run(output_names, {input_name: X_valid_array})

    # Print results for HDBSCAN here because we loose information when passing to cycle_and_conf1_df
    if len(pred_onnx) == 3:  # HDBSCAN onnx

        # print(pred_onnx[0], '\n')
        # print(pred_onnx[1], '\n')
        # print(pred_onnx[2], '\n')

        spread_ratio = pred_onnx[2][0, -1]  # ratio of testing spread with avg training clusters spread

        min_spread = 4  # biggest spread that corresponds to 100% confidence (smaller would also give 100% confidence because we clip)
        max_spread = 7  # spread that corresponds to 50% confidence
        slope = (0.5-1) / (max_spread - min_spread)
        intercept = 1 + (((1-0.5) * min_spread) / (max_spread - min_spread))
        confidence_level = np.clip((slope * spread_ratio + intercept), a_min=0, a_max=1)  # confidence level based on spread ratio

        med_wear = np.median(pred_onnx[2][:, 0])
        med_distance = np.median(pred_onnx[2][:, 1])
        print(f'HDBSCAN - SpreadRatio: {spread_ratio:.4f} - Distance metrics: ({med_wear:.4f}, {med_distance:.4f})')

    # Create df with cycle_id and conf1
    cycle_id_and_conf1 = np.column_stack((cycle_ids,  [confidence[1] for confidence in pred_onnx[1]]))
    cycle_and_conf1_df = pd.DataFrame(cycle_id_and_conf1, columns=['cycle_id', 'confidence_1'])

    return cycle_and_conf1_df


def skl2onnx_parser_castboost_classifier(scope, model, inputs, custom_parsers=None):
    options = scope.get_options(model, dict(zipmap=True))
    no_zipmap = isinstance(options["zipmap"], bool) and not options["zipmap"]

    alias = _get_sklearn_operator_name(type(model))
    this_operator = scope.declare_local_operator(alias, model)
    this_operator.inputs = inputs

    label_variable = scope.declare_local_variable("label", Int64TensorType())
    prob_dtype = guess_tensor_type(inputs[0].type)
    probability_tensor_variable = scope.declare_local_variable(
        "probabilities", prob_dtype
    )
    this_operator.outputs.append(label_variable)
    this_operator.outputs.append(probability_tensor_variable)
    probability_tensor = this_operator.outputs

    if no_zipmap:
        return probability_tensor

    return _apply_zipmap(
        options["zipmap"], scope, model, inputs[0].type, probability_tensor
    )


def skl2onnx_convert_catboost(scope, operator, container):
    """
    CatBoost returns an ONNX graph with a single node.
    This function adds it to the main graph.
    """
    onx = convert_to_onnx_object(operator.raw_operator)
    opsets = {d.domain: d.version for d in onx.opset_import}
    if "" in opsets and opsets[""] >= container.target_opset:
        raise RuntimeError("CatBoost uses an opset more recent than the target one.")
    if len(onx.graph.initializer) > 0 or len(onx.graph.sparse_initializer) > 0:
        raise NotImplementedError(
            "CatBoost returns a model initializers. This option is not implemented yet."
        )
    if (
        len(onx.graph.node) not in (1, 2)
        or not onx.graph.node[0].op_type.startswith("TreeEnsemble")
        or (len(onx.graph.node) == 2 and onx.graph.node[1].op_type != "ZipMap")
    ):
        types = ", ".join(map(lambda n: n.op_type, onx.graph.node))
        raise NotImplementedError(
            f"CatBoost returns {len(onx.graph.node)} != 1 (types={types}). "
            f"This option is not implemented yet."
        )
    node = onx.graph.node[0]
    atts = {}
    for att in node.attribute:
        atts[att.name] = get_attribute_value(att)
    container.add_node(
        node.op_type,
        [operator.inputs[0].full_name],
        [operator.outputs[0].full_name, operator.outputs[1].full_name],
        op_domain=node.domain,
        op_version=opsets.get(node.domain, None),
        **atts,
    )


update_registered_converter(
    CatBoostClassifier,
    "CatBoostCatBoostClassifier",
    calculate_linear_classifier_output_shapes,
    skl2onnx_convert_catboost,
    parser=skl2onnx_parser_castboost_classifier,
    options={"nocl": [True, False], "zipmap": [True, False, "columns"]},
)
