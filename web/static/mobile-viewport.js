(() => {
    const root = document.documentElement;
    const viewport = window.visualViewport;
    let frame;

    function syncViewport() {
        const unzoomed = viewport && Math.abs(viewport.scale - 1) < 0.01;
        const height = unzoomed ? viewport.height : window.innerHeight;
        const top = unzoomed ? Math.max(0, viewport.offsetTop) : 0;
        root.style.setProperty('--app-viewport-height', Math.round(height) + 'px');
        root.style.setProperty('--app-viewport-top', Math.round(top) + 'px');
        root.toggleAttribute('data-short-viewport', height < 420);
    }

    function scheduleSync() {
        cancelAnimationFrame(frame);
        frame = requestAnimationFrame(syncViewport);
    }

    window.addEventListener('resize', scheduleSync, { passive: true });
    window.addEventListener('orientationchange', scheduleSync, { passive: true });
    window.addEventListener('pageshow', scheduleSync);
    viewport?.addEventListener('resize', scheduleSync, { passive: true });
    viewport?.addEventListener('scroll', scheduleSync, { passive: true });
    screen.orientation?.addEventListener('change', scheduleSync);
    syncViewport();
})();
