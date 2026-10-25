# Récits nettoyés — narrative_only_v1

Dérivé du snapshot GOLD RQ1 `narrative_v3`. Chaque span explicitement présent dans `excluded_spans` a été retiré du texte brut; les unités factuelles sont conservées à l’identique et leurs offsets sont recalculés. Les originaux ne sont pas modifiés. Ce nettoyage n’infère aucune autre suppression.

`gold_snapshot.jsonl` contient tous les récits, y compris ceux sans unité. `data_<corpus>.csv` contient les unités factuelles, leur texte exact et le récit nettoyé. Les hashes et effectifs sont dans `narrative_only_manifest.json`.

Ce jeu convient aux expériences où l’entrée est le récit débarrassé des passages documentaires explicitement annotés. Il ne faut pas le présenter comme une annotation indépendante : les spans retirés proviennent des exclusions du GOLD existant.
