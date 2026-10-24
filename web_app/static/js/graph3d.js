/**
 * Heritage XAI - 3D Neuro-Symbolic Semantic Topology Graph (Three.js + GSAP)
 * --------------------------------------------------------------------------
 * Biểu diễn trực quan mạng lưới suy luận 3 tầng trong không gian 3D WebGL:
 *   - Tầng 1 (X = -14): 7 Trục MPEG-7 & Keyframes trích xuất
 *   - Tầng 2 (X = 0): Các Active Audio & Visual Concepts
 *   - Tầng 3 (X = +14): Các Luật Mờ Kích hoạt & Thể loại Dự đoán
 * Tương tác: Raycaster Hover Tooltip + GSAP Camera Fly-to khi click Concept Pills/Segments.
 * Tối ưu 60 FPS trên nền sáng (Editorial Light Mode).
 */

class SemanticGraph3D {
  constructor(containerId = "graph-canvas-container", tooltipId = "graph-tooltip") {
    this.container = document.getElementById(containerId);
    this.tooltip = document.getElementById(tooltipId);
    if (!this.container) {
      console.warn(`[SemanticGraph3D] Không tìm thấy phần tử #${containerId}`);
      return;
    }

    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.controls = null;

    // Node & Link data
    this.nodes = [];
    this.links = [];
    this.nodeMeshes = [];
    this.linkObjects = [];
    this.labelSprites = [];

    // Raycaster
    this.raycaster = new THREE.Raycaster();
    this.mouse = new THREE.Vector2(-999, -999);
    this.hoveredNode = null;

    // Loop & Observers
    this.animId = null;
    this.isVisible = true;

    this.initThree();
    this.initControls();
    this.initRaycaster();
    this.initObservers();
    this.startAnimationLoop();
  }

  initThree() {
    const width = this.container.clientWidth || 1000;
    const height = this.container.clientHeight || 420;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0xF9F8F4); // Màu nền giấy sáng đồng nhất

    this.camera = new THREE.PerspectiveCamera(40, width / height, 0.5, 200);
    this.camera.position.set(0, 10, 36);

    this.renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: false,
      powerPreference: "high-performance"
    });
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = false;

    this.container.innerHTML = "";
    this.container.appendChild(this.renderer.domElement);

    // Ánh sáng tự nhiên editorial
    const ambientLight = new THREE.AmbientLight(0xFFFFFF, 0.9);
    this.scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0xFFFBF0, 0.6);
    dirLight.position.set(20, 40, 20);
    this.scene.add(dirLight);

    // Thêm lưới tọa độ mặt sàn mảnh mờ
    const gridHelper = new THREE.GridHelper(40, 20, 0xE5E4DF, 0xF0EFEA);
    gridHelper.position.y = -10;
    this.scene.add(gridHelper);
  }

  initControls() {
    if (THREE.OrbitControls) {
      this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
      this.controls.enableDamping = true;
      this.controls.dampingFactor = 0.05;
      this.controls.maxPolarAngle = Math.PI / 2 + 0.1;
      this.controls.minDistance = 10;
      this.controls.maxDistance = 60;
      this.controls.target.set(0, 0, 0);
    }
  }

  initRaycaster() {
    this.container.addEventListener("mousemove", (e) => {
      const rect = this.container.getBoundingClientRect();
      this.mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      this.mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

      // Cập nhật vị trí tooltip
      if (this.tooltip && this.hoveredNode) {
        this.tooltip.style.left = `${e.clientX - rect.left + 16}px`;
        this.tooltip.style.top = `${e.clientY - rect.top + 16}px`;
      }
    });

    this.container.addEventListener("mouseleave", () => {
      this.mouse.x = -999;
      this.mouse.y = -999;
      this.hideTooltip();
    });

    this.container.addEventListener("click", () => {
      if (this.hoveredNode) {
        this.focusOnNode(this.hoveredNode.userData);
      }
    });
  }

  initObservers() {
    if (window.ResizeObserver) {
      const ro = new ResizeObserver((entries) => {
        for (const entry of entries) {
          const { width, height } = entry.contentRect;
          if (width > 0 && height > 0) {
            this.camera.aspect = width / height;
            this.camera.updateProjectionMatrix();
            this.renderer.setSize(width, height);
          }
        }
      });
      ro.observe(this.container);
    }

    if (window.IntersectionObserver) {
      const io = new IntersectionObserver((entries) => {
        this.isVisible = entries[0].isIntersecting;
      }, { threshold: 0.1 });
      io.observe(this.container);
    }
  }

  // ---------------------------------------------------------------------------
  // NẠP VÀ VẼ ĐỒ THỊ MỚI (RENDER GRAPH DATA)
  // ---------------------------------------------------------------------------

  loadGraphData(graphData) {
    if (!graphData || !graphData.nodes) return;

    this.clearGraph();

    this.nodes = graphData.nodes;
    this.links = graphData.links || [];

    const nodeMap = new Map();

    // 1. Tạo Node Spheres & Canvas Sprites
    for (const node of this.nodes) {
      const colorHex = parseInt(node.color.replace("#", "0x"), 16);

      let radius = 0.9;
      if (node.type === "verdict") radius = 1.6;
      else if (node.type.startsWith("concept")) radius = 1.1;
      else if (node.type === "rule") radius = 0.95;

      const geometry = new THREE.SphereGeometry(radius, 24, 24);
      const material = new THREE.MeshStandardMaterial({
        color: colorHex,
        roughness: 0.35,
        metalness: 0.15
      });

      const sphere = new THREE.Mesh(geometry, material);
      sphere.position.set(node.x, node.y, node.z);
      sphere.userData = node;

      // Hiệu ứng vành hào quang ngoài (Outer Ring)
      const ringGeo = new THREE.RingGeometry(radius * 1.15, radius * 1.25, 32);
      const ringMat = new THREE.MeshBasicMaterial({
        color: colorHex,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.4
      });
      const ring = new THREE.Mesh(ringGeo, ringMat);
      ring.lookAt(this.camera.position);
      sphere.add(ring);

      this.scene.add(sphere);
      this.nodeMeshes.push(sphere);
      nodeMap.set(node.id, sphere);

      // 2. Tạo nhãn văn bản (Sprite Label)
      const spriteLabel = this.createLabelSprite(node.name, node.color);
      spriteLabel.position.set(node.x, node.y + radius + 0.7, node.z);
      this.scene.add(spriteLabel);
      this.labelSprites.push(spriteLabel);
    }

    // 3. Tạo đường nối Splines uốn cong giữa các tầng (Layer Connectors)
    for (const link of this.links) {
      const srcNode = nodeMap.get(link.source);
      const tgtNode = nodeMap.get(link.target);

      if (srcNode && tgtNode) {
        const p1 = srcNode.position;
        const p2 = tgtNode.position;

        // Điểm uốn cong ở giữa
        const midX = (p1.x + p2.x) / 2;
        const midY = (p1.y + p2.y) / 2 + 1.2;
        const midZ = (p1.z + p2.z) / 2;
        const controlPoint = new THREE.Vector3(midX, midY, midZ);

        const curve = new THREE.QuadraticBezierCurve3(p1, controlPoint, p2);
        const points = curve.getPoints(24);
        const lineGeo = new THREE.BufferGeometry().setFromPoints(points);

        const lineMat = new THREE.LineBasicMaterial({
          color: 0xCAC7BD,
          transparent: true,
          opacity: 0.55,
          linewidth: 1.5
        });

        const line = new THREE.Line(lineGeo, lineMat);
        this.scene.add(line);
        this.linkObjects.push(line);
      }
    }

    // Đưa camera về góc nhìn toàn cảnh cân đối
    this.resetCameraView();
  }

  createLabelSprite(text, colorHexStr) {
    const canvas = document.createElement("canvas");
    canvas.width = 256;
    canvas.height = 64;
    const ctx = canvas.getContext("2d");

    ctx.clearRect(0, 0, 256, 64);

    // Nền tag nhãn mờ
    ctx.fillStyle = "rgba(255, 255, 255, 0.88)";
    ctx.beginPath();
    ctx.roundRect(8, 8, 240, 48, 8);
    ctx.fill();

    ctx.strokeStyle = "#E5E4DF";
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Dấu chấm màu
    ctx.fillStyle = colorHexStr;
    ctx.beginPath();
    ctx.arc(24, 32, 6, 0, Math.PI * 2);
    ctx.fill();

    // Văn bản
    ctx.font = "600 18px 'Plus Jakarta Sans', sans-serif";
    ctx.fillStyle = "#141413";
    ctx.fillText(text, 38, 38, 200);

    const texture = new THREE.Texture(canvas);
    texture.needsUpdate = true;

    const spriteMat = new THREE.SpriteMaterial({
      map: texture,
      transparent: true,
      depthTest: false
    });

    const sprite = new THREE.Sprite(spriteMat);
    sprite.scale.set(4.5, 1.15, 1);
    return sprite;
  }

  clearGraph() {
    for (const mesh of this.nodeMeshes) {
      mesh.geometry.dispose();
      mesh.material.dispose();
      this.scene.remove(mesh);
    }
    for (const label of this.labelSprites) {
      label.material.map.dispose();
      label.material.dispose();
      this.scene.remove(label);
    }
    for (const link of this.linkObjects) {
      link.geometry.dispose();
      link.material.dispose();
      this.scene.remove(link);
    }

    this.nodeMeshes = [];
    this.labelSprites = [];
    this.linkObjects = [];
    this.nodes = [];
    this.links = [];
  }

  // ---------------------------------------------------------------------------
  // TƯƠNG TÁC: GSAP CAMERA FLY-TO (2D DOM -> 3D SPACE)
  // ---------------------------------------------------------------------------

  /**
   * Bay mượt camera tới nút tương ứng khi người dùng click vào Concept Pill hoặc Timeline
   */
  focusOnNode(nodeData) {
    if (!nodeData || !window.gsap) return;

    const targetPos = new THREE.Vector3(nodeData.x, nodeData.y, nodeData.z);
    const cameraTargetPos = new THREE.Vector3(
      nodeData.x,
      nodeData.y + 1.5,
      nodeData.z + 12
    );

    // GSAP Motion Quy chuẩn: duration 1.0s, ease: power3.inOut
    gsap.to(this.camera.position, {
      x: cameraTargetPos.x,
      y: cameraTargetPos.y,
      z: cameraTargetPos.z,
      duration: 1.0,
      ease: "power3.inOut"
    });

    if (this.controls) {
      gsap.to(this.controls.target, {
        x: targetPos.x,
        y: targetPos.y,
        z: targetPos.z,
        duration: 1.0,
        ease: "power3.inOut"
      });
    }

    // Hiệu ứng xung nhịp rung nhẹ nút được chọn
    const mesh = this.nodeMeshes.find(m => m.userData.id === nodeData.id);
    if (mesh) {
      gsap.fromTo(mesh.scale,
        { x: 1.0, y: 1.0, z: 1.0 },
        { x: 1.35, y: 1.35, z: 1.35, duration: 0.35, yoyo: true, repeat: 1, ease: "power2.out" }
      );
    }
  }

  /**
   * Tìm kiếm theo concept_id (ví dụ CT_AUD_01) và kích hoạt fly-to
   */
  flyToConcept(conceptId) {
    const node = this.nodes.find(n => n.concept_id === conceptId || n.id === `node_concept_${conceptId}`);
    if (node) {
      this.focusOnNode(node);
    }
  }

  resetCameraView() {
    if (!window.gsap) {
      this.camera.position.set(0, 5, 34);
      if (this.controls) this.controls.target.set(0, 0, 0);
      return;
    }

    gsap.to(this.camera.position, {
      x: 0,
      y: 5,
      z: 34,
      duration: 1.2,
      ease: "power3.inOut"
    });

    if (this.controls) {
      gsap.to(this.controls.target, {
        x: 0,
        y: 0,
        z: 0,
        duration: 1.2,
        ease: "power3.inOut"
      });
    }
  }

  // ---------------------------------------------------------------------------
  // RAYCASTER HOVER & TOOLTIP
  // ---------------------------------------------------------------------------

  checkRaycaster() {
    if (this.mouse.x === -999) return;

    this.raycaster.setFromCamera(this.mouse, this.camera);
    const intersects = this.raycaster.intersectObjects(this.nodeMeshes, false);

    if (intersects.length > 0) {
      const topHit = intersects[0].object;
      if (this.hoveredNode !== topHit) {
        if (this.hoveredNode) this.resetNodeScale(this.hoveredNode);
        this.hoveredNode = topHit;
        this.hoverNodeScale(topHit);
        this.showTooltip(topHit.userData);
      }
    } else {
      if (this.hoveredNode) {
        this.resetNodeScale(this.hoveredNode);
        this.hoveredNode = null;
        this.hideTooltip();
      }
    }
  }

  hoverNodeScale(mesh) {
    if (window.gsap) {
      gsap.to(mesh.scale, { x: 1.2, y: 1.2, z: 1.2, duration: 0.25, ease: "power2.out" });
    } else {
      mesh.scale.set(1.2, 1.2, 1.2);
    }
  }

  resetNodeScale(mesh) {
    if (window.gsap) {
      gsap.to(mesh.scale, { x: 1.0, y: 1.0, z: 1.0, duration: 0.25, ease: "power2.out" });
    } else {
      mesh.scale.set(1.0, 1.0, 1.0);
    }
  }

  showTooltip(nodeData) {
    if (!this.tooltip) return;
    const titleEl = this.tooltip.querySelector(".graph-tooltip-title");
    const bodyEl = this.tooltip.querySelector(".graph-tooltip-body");

    if (titleEl) titleEl.innerText = nodeData.name;
    if (bodyEl) bodyEl.innerText = nodeData.details || "";

    this.tooltip.style.display = "block";
  }

  hideTooltip() {
    if (this.tooltip) {
      this.tooltip.style.display = "none";
    }
  }

  // ---------------------------------------------------------------------------
  // VÒNG LẶP RENDER CHÍNH (60 FPS)
  // ---------------------------------------------------------------------------

  startAnimationLoop() {
    const animate = () => {
      this.animId = requestAnimationFrame(animate);

      if (!this.isVisible) return;

      if (this.controls) this.controls.update();

      this.checkRaycaster();

      // Giữ cho các nhãn Sprite luôn hướng về camera
      for (const label of this.labelSprites) {
        label.quaternion.copy(this.camera.quaternion);
      }

      this.renderer.render(this.scene, this.camera);
    };

    animate();
  }

  dispose() {
    if (this.animId) cancelAnimationFrame(this.animId);
    this.clearGraph();
    if (this.controls) this.controls.dispose();
    if (this.renderer) this.renderer.dispose();
  }
}

// Global instance helper
window.SemanticGraph3D = SemanticGraph3D;
