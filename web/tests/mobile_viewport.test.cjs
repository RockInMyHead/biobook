const assert = require('node:assert/strict');
const fs = require('node:fs');
const test = require('node:test');
const vm = require('node:vm');

const source = fs.readFileSync(require.resolve('../static/mobile-viewport.js'), 'utf8');

function setupViewport() {
    const properties = new Map();
    const attributes = new Map();
    const callbacks = new Map();
    let frame = 0;
    const root = {
        style: { setProperty: (key, value) => properties.set(key, value) },
        toggleAttribute: (key, value) => attributes.set(key, value),
    };
    const viewport = Object.assign(new EventTarget(), { height: 480, offsetTop: 0, scale: 1 });
    const window = Object.assign(new EventTarget(), { innerHeight: 480, visualViewport: viewport });
    vm.runInNewContext(source, {
        document: { documentElement: root },
        window,
        screen: { orientation: new EventTarget() },
        requestAnimationFrame(callback) {
            callbacks.set(++frame, callback);
            return frame;
        },
        cancelAnimationFrame: id => callbacks.delete(id),
    });
    function flushFrame() {
        for (const callback of callbacks.values()) callback();
        callbacks.clear();
    }
    return { window, viewport, properties, attributes, flushFrame };
}

test('viewport resize and pan fit the editor to the visible keyboard area', () => {
    const { window, viewport, properties, attributes, flushFrame } = setupViewport();
    assert.equal(properties.get('--app-viewport-height'), '480px');

    window.innerHeight = 844;
    viewport.height = 325;
    viewport.offsetTop = 36;
    window.dispatchEvent(new Event('orientationchange'));
    viewport.dispatchEvent(new Event('resize'));
    viewport.dispatchEvent(new Event('scroll'));
    flushFrame();
    assert.equal(properties.get('--app-viewport-height'), '325px');
    assert.equal(properties.get('--app-viewport-top'), '36px');
    assert.equal(attributes.get('data-short-viewport'), true);

    viewport.height = 844;
    viewport.offsetTop = 0;
    viewport.dispatchEvent(new Event('resize'));
    flushFrame();
    assert.equal(properties.get('--app-viewport-height'), '844px');
    assert.equal(properties.get('--app-viewport-top'), '0px');
    assert.equal(attributes.get('data-short-viewport'), false);
});

test('pinch zoom leaves the application sized to the layout viewport', () => {
    const { window, viewport, properties, flushFrame } = setupViewport();
    window.innerHeight = 844;
    viewport.scale = 2;
    viewport.height = 420;
    viewport.offsetTop = 80;
    viewport.dispatchEvent(new Event('resize'));
    flushFrame();
    assert.equal(properties.get('--app-viewport-height'), '844px');
    assert.equal(properties.get('--app-viewport-top'), '0px');
});
