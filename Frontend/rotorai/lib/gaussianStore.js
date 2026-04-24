let viewerInstance = null;
let viewerCanvas = null;
let initPromise = null; // ← This is the key missing piece
let ready = false;

export const GaussianStore = {
  init(container) {
    // If already initializing or done, return the SAME promise
    if (initPromise) return initPromise;

    initPromise = (async () => {
      const GaussianSplats3D = await import('@mkkellogg/gaussian-splats-3d');

      viewerInstance = new GaussianSplats3D.Viewer({
        cameraUp: [0, 1, 0],
        initialCameraPosition: [2, -10, 1],
        initialCameraLookAt: [0, 0, 0],
        rootElement: container,
        sphericalHarmonicsDegree: 2,
      });

      viewerInstance.init();
      viewerInstance.renderer.toneMapping = 3;
      viewerInstance.renderer.toneMappingExposure = 1.2;
      viewerInstance.renderer.setPixelRatio(window.devicePixelRatio);
      viewerCanvas = viewerInstance.renderer.domElement;

      viewerCanvas.style.width = '100%';
      viewerCanvas.style.height = '100%';
      viewerCanvas.style.position = 'absolute';
      viewerCanvas.style.top = '0';
      viewerCanvas.style.left = '0';

      await viewerInstance.addSplatScene('/models/best_motor.ply', {
        progressiveLoad: false,
        streamView: true,
      });

      viewerInstance.start();
      viewerInstance.renderer.setSize = ((original) =>
        function (w, h, updateStyle) {
          const container = viewerCanvas?.parentNode;
          if (container) {
            const rect = container.getBoundingClientRect();
            return original.call(this, rect.width, rect.height, updateStyle);
          }
          return original.call(this, w, h, updateStyle);
        })(viewerInstance.renderer.setSize.bind(viewerInstance.renderer));

      ready = true;
    })();

    return initPromise;
  },

  attach(container) {
    if (viewerCanvas && container && !viewerCanvas.parentNode) {
      container.appendChild(viewerCanvas);

      // Force canvas to match container exactly
      const rect = container.getBoundingClientRect();
      viewerInstance.renderer.setSize(rect.width, rect.height, false);
      viewerCanvas.style.width = '100%';
      viewerCanvas.style.height = '100%';
      viewerCanvas.style.position = 'absolute';
      viewerCanvas.style.top = '0';
      viewerCanvas.style.left = '0';

      // Suppress the resize event from propagating to the viewer's own handler
      viewerInstance.updateViewDimensions?.();
    }
  },

  detach() {
    if (viewerCanvas?.parentNode) {
      viewerCanvas.parentNode.removeChild(viewerCanvas);
    }
    try {
      viewerInstance?.dispose?.();
    } catch (e) {}
    viewerInstance = null;
    viewerCanvas = null;
    initPromise = null;
    ready = false;
  },

  isReady() {
    return ready;
  },
};
