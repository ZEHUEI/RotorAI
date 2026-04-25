import * as THREE from 'three';

let viewerInstance = null;
let viewerCanvas = null;
let initPromise = null;
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
      const axesHelper = new THREE.AxesHelper(5);
      //red: X, green: Y, Blue: Z
      viewerInstance.threeScene.add(axesHelper);
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

      // Load and draw YOLO detection boxes
      const response = await fetch(
        '/models/clustered_detections_circle_best.json'
      );
      const detections = await response.json();

      const classColors = {
        rust: 0x00ff00, // Green
        crack: 0x00e5ff, // Cyan
      };

      detections.forEach((det) => {
        // 2. Create Box Geometry using the [width, height, depth] from our Python script
        const geometry = new THREE.BoxGeometry(
          det.size[0],
          det.size[1],
          det.size[2]
        );

        // 3. Transparent Material for that 'High-Res Engine' look
        const color = classColors[det.type] || 0xffff00;
        const material = new THREE.MeshBasicMaterial({
          color: color,
          transparent: true,
          opacity: 0.5,
          depthWrite: false, // Prevents boxes from cutting through each other
        });

        const box = new THREE.Mesh(geometry, material);

        //debug
        // const x = det.position[0] - 0.05;
        // const y = det.position[1] + 0.05;
        // const z = det.position[2] + 1.0;

        //real
        const x = det.position[0] - 0.0;
        const y = det.position[1] + 0.05;
        const z = det.position[2] - 0.0;

        box.rotation.y = 0.0;
        box.rotation.z = -0.0;

        // 4. Set Position from our [x, y, z] mapping
        box.position.set(x, y, z);

        // 5. Add Wireframe Outline
        const edges = new THREE.EdgesGeometry(geometry);
        const line = new THREE.LineSegments(
          edges,
          new THREE.LineBasicMaterial({ color: color, linewidth: 2 })
        );
        box.add(line);

        // 6. Add to the Splat Scene
        viewerInstance.threeScene.add(box);
      });

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
