import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from web_app.inference_engine import HeritageInferenceEngine

def main():
    engine = HeritageInferenceEngine()
    print("=" * 65)
    print("KIEM TRA TOAN DIEN: LIEN KET ARROWS 100% & PHUC HOI LUAT INTERVENTION")
    print("=" * 65)

    test_files = [
        ('dataset/video/catru/catru_tayhohoaico_03.mp4', 'video', 'Ca trù'),
        ('dataset/video/cheo/cheo_luubinhduongle_08.mp4', 'video', 'Chèo'),
        ('dataset/video/cailuong/cailuong_doicoluu_06.mp4', 'video', 'Cải lương'),
        ('dataset/video/quanho/quanho_caytrucxinh.mp4', 'video', 'Quan họ')
    ]

    for file_path, mode, exp_genre in test_files:
        print(f"\n--- [1] Kiem tra Nap du lieu moi: {file_path} ({exp_genre}) ---")
        if not Path(file_path).exists():
            print(f"  [SKIP] Khong tim thay {file_path}")
            continue

        res = engine.analyze(file_path, mode=mode)
        em = res['evidence_map']
        nodes = em['nodes']
        links = em['links']
        rules = res['triggered_rules']

        concept_nodes = [n for n in nodes if n['category'] == 'concept']
        rule_nodes = [n for n in nodes if n['category'] == 'rule']

        print(f"  So luong Concepts: {len(concept_nodes)} | So luong Rules: {len(rule_nodes)} (triggered_rules: {len(rules)})")
        assert len(rule_nodes) >= 2, f"Loi: Thieu luat! Chi co {len(rule_nodes)} luat, yeu cau >= 2 luat da phuong thuc."

        # Kiem tra 100% concept nodes co mui ten ra (outgoing link toi rule node)
        unlinked_concepts = []
        for cn in concept_nodes:
            has_link = any(l['source'] == cn['id'] and str(l['target']).startswith('node_rule_') for l in links)
            if not has_link:
                unlinked_concepts.append(cn['id'])

        if unlinked_concepts:
            print(f"  [THAT BAI] Cac concept khong co mui ten toi rule: {unlinked_concepts}")
            assert False, f"Van con {len(unlinked_concepts)} concept bi mo coi khong co mui ten toi Tang 4!"
        else:
            print(f"  [THANH CONG] 100% concept nodes ({len(concept_nodes)}/{len(concept_nodes)}) deu co mui ten toi Tang 4 (Rules)!")

        # Kiem tra used_in_rules tren concept
        for cn in concept_nodes:
            assert len(cn.get('used_in_rules', [])) > 0, f"Concept {cn['id']} used_in_rules bi rong!"

        # [2] Kiem tra Intervention: Toggle OFF -> Toggle ON
        print(f"\n--- [2] Kiem tra Can thiep CBM & Phuc hoi 100% luat ({exp_genre}) ---")
        base_probs = res['probs']
        base_rules = res['triggered_rules']
        active_concepts = res['active_concepts']
        concept_to_toggle = active_concepts[0]['concept_id']

        print(f"  [Buoc 2.1] Toggle OFF concept: {concept_to_toggle}")
        intervene_off = engine.intervene(
            current_probs=base_probs,
            toggled_concept_id=concept_to_toggle,
            is_enabled=False,
            all_concepts=active_concepts,
            base_probs=base_probs,
            base_rules=base_rules
        )
        rules_off = intervene_off['triggered_rules']
        em_off = intervene_off['evidence_map']
        print(f"  Ket qua khi OFF: So luat = {len(rules_off)} | Conf = {intervene_off['confidence']}%")
        assert len(rules_off) >= 1, "Loi: Khong co luat nao sau khi can thiep OFF!"

        # Kiem tra 100% concept lien ket trong evidence map khi OFF
        c_nodes_off = [n for n in em_off['nodes'] if n['category'] == 'concept']
        for cn in c_nodes_off:
            has_link = any(l['source'] == cn['id'] and str(l['target']).startswith('node_rule_') for l in em_off['links'])
            assert has_link, f"Concept {cn['id']} bi mat link khi OFF!"

        print(f"  [Buoc 2.2] Toggle ON lai concept: {concept_to_toggle} (Phuc hoi trang thai ban dau)")
        intervene_on = engine.intervene(
            current_probs=intervene_off['probs'],
            toggled_concept_id=concept_to_toggle,
            is_enabled=True,
            all_concepts=intervene_off['all_concepts'] if 'all_concepts' in intervene_off else active_concepts,
            base_probs=base_probs,
            base_rules=base_rules
        )
        rules_on = intervene_on['triggered_rules']
        em_on = intervene_on['evidence_map']

        print(f"  So luat ban dau: {len(base_rules)} | So luat sau khi Toggle ON lai: {len(rules_on)}")
        assert len(rules_on) == len(base_rules), f"Loi: So luong luat khong duoc phuc hoi! Truoc={len(base_rules)}, Sau={len(rules_on)}"
        for idx in range(len(base_rules)):
            assert rules_on[idx]['rule_text'] == base_rules[idx]['rule_text'], f"Loi noi dung luat #{idx+1} khong khop!"

        print(f"  [THANH CONG] Phuc hoi 100% tat ca {len(base_rules)} luat ban dau chinh xac tuyet doi!")

    print("\n" + "=" * 65)
    print("HOAN TAT! TAT CA CAC DIEU KIEN KIEM TRA DEU DAT 100%!")
    print("=" * 65)

if __name__ == '__main__':
    main()
