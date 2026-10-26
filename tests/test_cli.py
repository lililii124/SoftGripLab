import json

import numpy as np
import pytest

from softgriplab.cli import main
from softgriplab.demo import analytic_ramp_force
from softgriplab.identification import CompressionTrace
from softgriplab.materials import Geometry, StandardLinearSolid, ramp_hold


def test_identify_cli_accepts_external_csv(tmp_path):
    t = np.linspace(0, 3, 151)
    material = StandardLinearSolid(25000, 50000, .4)
    csv = tmp_path / "measured.csv"
    CompressionTrace(t, ramp_hold(t, .001, .2),
                     analytic_ramp_force(material, t, .001, .2, Geometry())).to_csv(csv)
    output = tmp_path / "fit.json"
    main(["identify", str(csv), "--thickness-m", ".02", "--area-m2", ".0004",
          "--output", str(output)])
    report = json.loads(output.read_text())
    assert abs(report["material"]["equilibrium_pa"] / 25000 - 1) < .001


def test_malformed_csv_fails_cleanly(tmp_path, capsys):
    csv = tmp_path / "bad.csv"
    csv.write_text("time,displacement,force\n0,0,0\n")
    with pytest.raises(SystemExit) as error:
        main(["identify", str(csv), "--thickness-m", ".02", "--area-m2", ".0004"])
    assert error.value.code == 2
    assert "CSV requires" in capsys.readouterr().err
