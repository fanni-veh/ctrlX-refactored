// When the app is served behind a sub-path proxy (e.g. ctrlX at /mind),
// all HTMX requests must include that prefix. The prefix is set server-side
// via the ASGI root_path and exposed as <meta name="app-root">.
(function () {
    var meta = document.querySelector('meta[name="app-root"]');
    var prefix = meta ? meta.getAttribute('content') : '';

    // Only activate if there is a non-empty root path
    if (!prefix) return;

    document.addEventListener('htmx:configRequest', function (evt) {
        var path = evt.detail.path;
        if (path && path.charAt(0) === '/' && path.indexOf(prefix + '/') !== 0 && path !== prefix) {
            evt.detail.path = prefix + path;
        }
    });
})();
