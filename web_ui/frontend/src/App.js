import React, { useState, useRef, useCallback } from 'react';
import { Canvas, useLoader, useThree } from '@react-three/fiber';
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader';
import * as THREE from 'three';
import axios from 'axios';
import './App.css';

const API = 'http://127.0.0.1:5000/api';

// 3D Point Cloud Component
function PointCloud({ url, position, rotation, scale }) {
  const geometry = useLoader(PLYLoader, url);
  const ref = useRef();

  React.useMemo(() => {
    geometry.computeBoundingBox();
    const box = geometry.boundingBox;
    const cx = (box.min.x + box.max.x) / 2;
    const cy = (box.min.y + box.max.y) / 2;
    const minZ = box.min.z;
    geometry.translate(-cx, -cy, -minZ);
  }, [geometry]);

  return (
    <group position={position} rotation={rotation} scale={[scale, scale, scale]}>
      <points ref={ref}>
        <bufferGeometry attach="geometry" {...geometry} />
        <pointsMaterial attach="material" size={0.002} vertexColors sizeAttenuation />
      </points>
    </group>
  );
}

// Left-drag = move object, right-drag = rotate object in place, scroll = zoom
function DragHandler({ onMove, onRotate, onZoom }) {
  const { camera, gl } = useThree();
  const dragging = useRef(false);
  const rotating = useRef(false);
  const lastMouse = useRef({ x: 0, y: 0 });

  React.useEffect(() => {
    const raycaster = new THREE.Raycaster();

    const onDown = (e) => {
      e.preventDefault();
      lastMouse.current = { x: e.clientX, y: e.clientY };
      if (e.button === 0) {
        dragging.current = true;
        gl.domElement.style.cursor = 'grabbing';
      } else if (e.button === 2) {
        rotating.current = true;
        gl.domElement.style.cursor = 'crosshair';
      }
    };

    const onMoveEvt = (e) => {
      if (dragging.current) {
        const rect = gl.domElement.getBoundingClientRect();
        const mouse = new THREE.Vector2(
          ((e.clientX - rect.left) / rect.width) * 2 - 1,
          -((e.clientY - rect.top) / rect.height) * 2 + 1
        );
        raycaster.setFromCamera(mouse, camera);
        const plane = new THREE.Plane(new THREE.Vector3(0, 0, 1), -0.3);
        const pt = new THREE.Vector3();
        raycaster.ray.intersectPlane(plane, pt);
        if (pt) onMove([pt.x, pt.y, 0]);
      }
      if (rotating.current) {
        const dx = e.clientX - lastMouse.current.x;
        const dy = e.clientY - lastMouse.current.y;
        onRotate(dx * 0.5, dy * 0.5);
        lastMouse.current = { x: e.clientX, y: e.clientY };
      }
    };

    const onUp = () => {
      dragging.current = false;
      rotating.current = false;
      gl.domElement.style.cursor = 'default';
    };

    const onWheel = (e) => {
      e.preventDefault();
      onZoom(e.deltaY > 0 ? 1.05 : 0.95);
    };

    const onContext = (e) => e.preventDefault();

    const el = gl.domElement;
    el.addEventListener('pointerdown', onDown);
    el.addEventListener('pointermove', onMoveEvt);
    el.addEventListener('pointerup', onUp);
    el.addEventListener('wheel', onWheel, { passive: false });
    el.addEventListener('contextmenu', onContext);
    return () => {
      el.removeEventListener('pointerdown', onDown);
      el.removeEventListener('pointermove', onMoveEvt);
      el.removeEventListener('pointerup', onUp);
      el.removeEventListener('wheel', onWheel);
      el.removeEventListener('contextmenu', onContext);
    };
  }, [camera, gl, onMove, onRotate, onZoom]);

  return null;
}

function App() {
  const [plyId, setPlyId] = useState(null);
  const [plyUrl, setPlyUrl] = useState(null);
  const [plyName, setPlyName] = useState('');
  const [sceneId, setSceneId] = useState(null);
  const [sceneExt, setSceneExt] = useState('.jpg');
  const [sceneName, setSceneName] = useState('');
  const [scenePreview, setScenePreview] = useState(null);
  const [renderedUrl, setRenderedUrl] = useState(null);
  const [maskUrl, setMaskUrl] = useState(null);
  const [compositeUrl, setCompositeUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [objPosition, setObjPosition] = useState([0, 0, 0]);
  const [objRotation, setObjRotation] = useState([0, 0, 0]);
  const [objScale, setObjScale] = useState(1);
  const controlsRef = useRef();

  const handleMove = useCallback((pos) => {
    setObjPosition(pos);
  }, []);

  const handleRotate = useCallback((dx, dy) => {
    setObjRotation(prev => [
      prev[0] + THREE.MathUtils.degToRad(dy),
      prev[1],
      prev[2] + THREE.MathUtils.degToRad(dx),
    ]);
  }, []);

  const handleZoom = useCallback((factor) => {
    setObjScale(prev => Math.max(0.1, Math.min(5, prev * factor)));
  }, []);

  const handlePlyUpload = useCallback(async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const form = new FormData();
    form.append('file', file);
    const res = await axios.post(`${API}/upload-ply`, form);
    setPlyId(res.data.id);
    setPlyName(res.data.filename);
    setPlyUrl(URL.createObjectURL(file));
  }, []);

  const handleSceneUpload = useCallback(async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const form = new FormData();
    form.append('file', file);
    const res = await axios.post(`${API}/upload-scene`, form);
    setSceneId(res.data.id);
    setSceneExt(res.data.ext);
    setSceneName(res.data.filename);
    setScenePreview(URL.createObjectURL(file));
  }, []);

  const handleExport = useCallback(async () => {
    if (!plyId) return;
    setLoading(true);
    try {
      const azimuth = controlsRef.current
        ? THREE.MathUtils.radToDeg(controlsRef.current.getAzimuthalAngle())
        : 0;

      const res = await axios.post(`${API}/render`, {
        ply_id: plyId,
        yaw: azimuth,
      });
      const oid = res.data.output_id;
      setRenderedUrl(`${API}/file/${oid}/rendered.png?t=${Date.now()}`);
      setMaskUrl(`${API}/file/${oid}/mask.png?t=${Date.now()}`);
    } catch (err) {
      alert('Export failed: ' + (err.response?.data?.error || err.message));
    }
    setLoading(false);
  }, [plyId, sceneId, sceneExt]);

  return (
    <div className="app">
      <h1>MetaSam3D Viewer</h1>

      {/* Upload Section */}
      <div className="section">
        <div className="upload-row">
          <div className="upload-box">
            <label>Upload .PLY Model</label>
            <input type="file" accept=".ply" onChange={handlePlyUpload} />
            {plyName && <span className="filename">{plyName}</span>}
          </div>
          <div className="upload-box">
            <label>Upload Background Scene</label>
            <input type="file" accept="image/*" onChange={handleSceneUpload} />
            {sceneName && <span className="filename">{sceneName}</span>}
          </div>
        </div>
      </div>

      {/* Main Split View */}
      {(plyUrl || scenePreview) && (
        <div className="section">
          <div className="split-view">
            {/* Left: Input Scene */}
            <div className="split-panel">
              <h2>Input Scene</h2>
              {scenePreview ? (
                <img src={scenePreview} alt="input scene" className="scene-img" />
              ) : (
                <div className="placeholder">Upload a background scene</div>
              )}
            </div>

            {/* Right: 3D Viewer with scene as CSS background */}
            <div className="split-panel">
              <h2>Output Preview — Drag to rotate freely</h2>
              <div
                className="viewer-container"
                style={scenePreview ? {
                  backgroundImage: `url(${scenePreview})`,
                  backgroundSize: 'cover',
                  backgroundPosition: 'center',
                } : {}}
              >
                <Canvas
                  camera={{ position: [0, 1.2, 0.4], up: [0, 0, 1], fov: 50 }}
                  gl={{ alpha: true }}
                  style={{ background: 'transparent' }}
                >
                  <ambientLight intensity={1.0} />
                  <directionalLight position={[5, 5, 5]} intensity={0.6} />
                  <directionalLight position={[-5, -5, 3]} intensity={0.3} />

                  {plyUrl && (
                    <React.Suspense fallback={null}>
                      <PointCloud url={plyUrl} position={objPosition} rotation={objRotation} scale={objScale} />
                    </React.Suspense>
                  )}

                  <DragHandler onMove={handleMove} onRotate={handleRotate} onZoom={handleZoom} />
                </Canvas>
              </div>
              <p className="hint">
                Left-drag: move object | Right-drag: rotate | Scroll: zoom
              </p>
            </div>
          </div>

          {/* Export Button */}
          <div className="export-bar">
            <button onClick={handleExport} disabled={loading || !plyId} className="export-btn">
              {loading ? 'Exporting...' : 'Export Rendered + Mask'}
            </button>
          </div>
        </div>
      )}

      {/* Export Results */}
      {renderedUrl && (
        <div className="section">
          <h2>Export Results</h2>
          <div className="result-row">
            <div>
              <h3>Rendered</h3>
              <img src={renderedUrl} alt="rendered" />
            </div>
            <div>
              <h3>Mask</h3>
              <img src={maskUrl} alt="mask" />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;