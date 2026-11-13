

def test_verificar_exige_eval_vigente_y_umbral(tmp_path):
    """Issue #40: sin reporte, con casos viejos o bajo el umbral, la verificación falla."""
    import hashlib

    from credit_lab.evals import verificar

    (tmp_path / "casos.json").write_text("[]", encoding="utf-8")
    huella = hashlib.sha256(b"[]").hexdigest()[:10]
    ruta = tmp_path / "m" / "v9"
    assert verificar("v9", "m", tmp_path)
    ruta.mkdir(parents=True)
    (ruta / "reporte.md").write_text(f"casos.json {huella}\n| concordancia | 79% |\n", encoding="utf-8")
    assert any("umbral" in p for p in verificar("v9", "m", tmp_path))
    (ruta / "reporte.md").write_text(f"casos.json {huella}\n| concordancia | 97% |\n", encoding="utf-8")
    assert verificar("v9", "m", tmp_path) == []
