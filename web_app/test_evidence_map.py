import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from web_app.inference_engine import HeritageInferenceEngine

def run_tests():
    engine = HeritageInferenceEngine()
    test_files = [
        ('dataset/video/catru/catru_tayhohoaico_03.mp4', 'video', 'Ca trù'),
        ('dataset/video/cheo/cheo_luubinhduongle_08.mp4', 'video', 'Chèo'),
        ('dataset/video/cailuong/cailuong_doicoluu_06.mp4', 'video', 'Cải lương'),
        ('dataset/video/quanho/quanho_caytrucxinh.mp4', 'video', 'Quan họ')
    ]

    print("=" * 60)
    print("BAT DAU KIEM TRA TOAN DIEN MULTIMODAL EVIDENCE MAP")
    print("=" * 60)

    for file_path, mode, exp_genre in test_files:
        print(f"\n--- Kiem tra file: {file_path} ({mode}) ---")
        if not Path(file_path).exists():
            print(f"  [SKIP] Khong tim thay file {file_path}")
            continue

        res = engine.analyze(file_path, mode=mode)
        em = res.get('evidence_map')
        assert em is not None, "evidence_map is None!"
        nodes = em['nodes']
        links = em['links']
        node_ids = {n['id'] for n in nodes}
        
        # 1. Kiem tra tinh duy nhat cua Node IDs
        assert len(nodes) == len(node_ids), f"Trung lap Node ID: {len(nodes)} vs {len(node_ids)}"

        # 2. Kiem tra tinh toan ven cua tat ca cac Links (source va target ton tai)
        for l in links:
            assert l['source'] in node_ids, f"Link source '{l['source']}' khong ton tai trong nodes!"
            assert l['target'] in node_ids, f"Link target '{l['target']}' khong ton tai trong nodes!"

        # 3. Kiem tra su ket noi giua tat ca cac tang (0 -> 1 -> 2 -> 3 -> 4)
        layer_map = {n['id']: n['layer'] for n in nodes}
        l0_to_l1 = [l for l in links if layer_map.get(l['source']) == 0 and layer_map.get(l['target']) == 1]
        l1_to_l2 = [l for l in links if layer_map.get(l['source']) == 1 and layer_map.get(l['target']) == 2]
        l2_to_l3 = [l for l in links if layer_map.get(l['source']) == 2 and layer_map.get(l['target']) == 3]
        l3_to_l4 = [l for l in links if layer_map.get(l['source']) == 3 and layer_map.get(l['target']) == 4]

        print(f"  Tong so Nodes: {len(nodes)} | Tong so Links: {len(links)}")
        print(f"  Links T0->T1 (Dau vao -> Dac trung): {len(l0_to_l1)}")
        print(f"  Links T1->T2 (Dac trung -> Khai niem): {len(l1_to_l2)}")
        print(f"  Links T2->T3 (Khai niem -> Luat): {len(l2_to_l3)}")
        print(f"  Links T3->T4 (Luat -> Phan quyet): {len(l3_to_l4)}")

        assert len(l0_to_l1) > 0, "Loi: Khong co link T0 -> T1!"
        assert len(l1_to_l2) > 0, "Loi: Khong co link T1 -> T2!"
        assert len(l2_to_l3) > 0, "Loi: Khong co link T2 -> T3 (Khai niem -> Luat)!"
        assert len(l3_to_l4) > 0, "Loi: Khong co link T3 -> T4 (Luat -> Phan quyet)!"

        # 4. Kiem tra node Audio CNN
        cnn_node = next((n for n in nodes if n['id'] == 'node_feat_audio_cnn'), None)
        assert cnn_node is not None, "Loi: Thieu node Audio CNN (node_feat_audio_cnn) o Tang 1!"
        print(f"  Node Audio CNN: {cnn_node['name']} ({cnn_node['badge']}) -> PASS")

        # 5. Kiem tra Timestamps tren cac Concept nodes
        concept_nodes = [n for n in nodes if n['category'] == 'concept']
        for cn in concept_nodes:
            ts = cn.get('timestamps')
            assert ts is not None and len(ts) > 0, f"Concept node {cn['id']} khong co timestamps!"
        print(f"  Timestamps tren {len(concept_nodes)} concept nodes deu hop le -> PASS")

        # 6. Kiem tra Rule nodes co input_concept_ids hop le
        rule_nodes = [n for n in nodes if n['category'] == 'rule']
        for rn in rule_nodes:
            inp = rn.get('input_concept_ids', [])
            assert len(inp) > 0, f"Rule node {rn['id']} khong co input_concept_ids!"
            for cid in inp:
                assert cid in node_ids, f"Rule input concept {cid} khong co trong nodes!"
        print(f"  Input concepts tren {len(rule_nodes)} rule nodes deu hop le -> PASS")

    # 7. Kiem tra che do Audio-only
    print("\n--- Kiem tra Che do Audio-only (Thieu kenh thi giac) ---")
    res_audio = engine._build_evidence_map(
        genre="Ca trù",
        confidence=0.72,
        chunks=[],
        concepts=[
            {'concept_id': 'CT_AUD_01', 'aspect': 'nhac_cu', 'concept_sentence': 'Đàn đáy', 'type': 'audio', 'timestamps': ['00:00 - 00:04']},
            {'concept_id': 'CT_AUD_02', 'aspect': 'tiet_tau', 'concept_sentence': 'Phách', 'type': 'audio', 'timestamps': ['00:00 - 00:04']}
        ],
        rules=[{
            'rule_text': 'dan_day == 1 -> Ca tru',
            'consequent': 'Ca trù',
            'confidence': 0.85,
            'summary_text': 'Luat dan day',
            'clauses': ['dan_day']
        }],
        mode="audio",
        file_name="test_audio.wav"
    )
    assert res_audio['has_visual'] is False, "has_visual phai la False trong che do audio!"
    audio_nodes = res_audio['nodes']
    assert not any(n['modality'] == 'visual' for n in audio_nodes), "Che do audio khong duoc chua visual nodes!"
    print(f"  Audio-only: has_visual={res_audio['has_visual']}, Nodes={len(audio_nodes)}, Links={len(res_audio['links'])} -> PASS")

    # 8. Kiem tra Can thiep CBM Phản sự thực (/api/intervene)
    print("\n--- Kiem tra Can thiep CBM Phản sự thực (/api/intervene) ---")
    sample_res = engine.analyze('dataset/video/catru/catru_tayhohoaico_03.mp4', mode='video')
    int_res = engine.intervene(
        current_probs=sample_res['probs'],
        toggled_concept_id='CT_AUD_05',
        is_enabled=False,
        all_concepts=sample_res['active_concepts'],
        base_probs=sample_res['probs']
    )
    assert 'evidence_map' in int_res, "Ket qua can thiep thieu evidence_map!"
    int_em = int_res['evidence_map']
    assert len(int_em['nodes']) > 0 and len(int_em['links']) > 0, "evidence_map sau can thiep rong!"
    print(f"  Can thiep CBM thanh cong: Nodes={len(int_em['nodes'])}, Links={len(int_em['links'])} -> PASS")

    print("\n" + "=" * 60)
    print("TAT CA CAC KIEM TRA CHUC NANG MULTIMODAL EVIDENCE MAP DEU DAT!")
    print("=" * 60)

if __name__ == '__main__':
    run_tests()
