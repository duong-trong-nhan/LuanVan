# -*- coding: utf-8 -*-
"""
Test script for Heritage XAI Backend API
"""

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi.testclient import TestClient
from web_app.app import app

def run_tests():
    client = TestClient(app)

    print("=" * 60)
    print("1. Kiểm tra GET / (Index HTML):")
    res = client.get("/")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    assert "Hệ thống Phân lớp" in res.text, "Index HTML missing title"
    print("  -> PASS: Index HTML load thanh cong!")

    print("\n2. Kiểm tra GET /api/health:")
    res = client.get("/api/health")
    assert res.status_code == 200
    print("  -> PASS:", res.json())

    print("\n3. Kiểm tra GET /api/sample_files:")
    res = client.get("/api/sample_files")
    assert res.status_code == 200
    samples = res.json().get("samples", [])
    assert len(samples) >= 6, f"Expected at least 6 samples, got {len(samples)}"
    print(f"  -> PASS: Lay thanh cong {len(samples)} mau representative!")

    print("\n4. Kiểm tra POST /api/analyze_sample (Mẫu Ca trù):")
    sample_catru = samples[0]
    res = client.post("/api/analyze_sample", json={
        "sample_path": sample_catru["file_path"],
        "mode": "video"
    })
    assert res.status_code == 200, f"Analysis failed: {res.text}"
    data = res.json()
    assert data["prediction"] == "Ca trù", f"Expected Ca trù, got {data['prediction']}"
    assert "chunks" in data and len(data["chunks"]) > 0
    assert "merged_segments" in data and len(data["merged_segments"]) > 0
    assert "timeline_series" in data and len(data["timeline_series"]) > 0
    assert "modalities" in data and "audio" in data["modalities"] and "fused" in data["modalities"]
    assert "active_concepts" in data and len(data["active_concepts"]) > 0
    assert "graph_3d" in data and len(data["graph_3d"]["nodes"]) > 0
    print(f"  -> PASS: Du doan {data['prediction']}, Tin cay: {data['confidence']}%, Chunks: {len(data['chunks'])}, Merged: {len(data['merged_segments'])}, Timeline: {len(data['timeline_series'])}, Modal: {list(data['modalities'].keys())}")

    print("\n5. Kiểm tra POST /api/intervene (Can thiep CBM):")
    cbm_req = {
        "current_probs": data["probs"],
        "concept_id": "CT_AUD_01",
        "is_enabled": False,
        "all_concepts": data["active_concepts"]
    }
    res = client.post("/api/intervene", json=cbm_req)
    assert res.status_code == 200
    cbm_res = res.json()
    print("  -> PASS: Can thiep thanh cong! Xac suat sau can thiep:", cbm_res["probs"])

    print("\n=" * 60)
    print("TAT CA CAC KIEM THU BACKEND DEU DAT CHUAN!")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
