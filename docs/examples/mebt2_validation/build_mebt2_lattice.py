"""Build the Cheetah LatticeJSON model for the J-PARC muon MEBT-II.

The source values mirror mebt2_v4_324MHz.in (2026-09-18 revision).

Important modelling choices
---------------------------
* The lattice coordinate s=0 is the IH-DTL1 exit / MEBT-II entrance.
* S1/S2 are vertical steering correctors. Their GPT rectmagnet fields are
  converted to Cheetah kick angles at the GPT design rigidity.
* GPT quadrupole gradients [T/m] are converted to Cheetah normalized k1
  [1/m^2] at the GPT design rigidity (gamma=1.040294013).
* The 324 MHz buncher mechanical envelope is 74 mm, but the GPT field-map
  effective length is 15.68433 mm. Cheetah therefore uses:
      side drift + active Cavity(Leff) + side drift
  while preserving the 74 mm survey envelope.
* For positive muons, the current Cheetah Cavity convention is compensated
  by CAVITY_VOLTAGE_SIGN=-1. This keeps GPT and Cheetah phase numbers equal
  initially. Validate/calibrate this sign/phase relation with GPT scans.
* BM1/BM2 are represented as sector Dipoles with +3.5 deg reference bend.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

MUON_MASS_MEV = 105.6583755
DESIGN_GAMMA = 1.040294013
DESIGN_KINETIC_ENERGY_MEV = (DESIGN_GAMMA - 1.0) * MUON_MASS_MEV
DESIGN_P0C_MEV = math.sqrt(
    (DESIGN_GAMMA * MUON_MASS_MEV) ** 2 - MUON_MASS_MEV**2
)
DESIGN_P0C_GEV = DESIGN_P0C_MEV / 1000.0
BRHO_TM = DESIGN_P0C_GEV / 0.299792458

RF_FREQUENCY_HZ = 324.0e6
QUAD_LENGTH_M = 0.120
CORRECTOR_LENGTH_M = 0.030
CAVITY_MECHANICAL_LENGTH_M = 0.074
CAVITY_EFFECTIVE_LENGTH_M = 0.01568433
CAVITY_SIDE_DRIFT_M = (
    CAVITY_MECHANICAL_LENGTH_M - CAVITY_EFFECTIVE_LENGTH_M
) / 2.0

BEND_RADIUS_M = 2.30
BEND_ANGLE_DEG = 3.5
BEND_ANGLE_RAD = math.radians(BEND_ANGLE_DEG)
BEND_LENGTH_M = BEND_RADIUS_M * BEND_ANGLE_RAD
BEND_GAP_M = 0.080

# Cheetah's current Cavity convention uses a sign that is electron-oriented.
# For mu+, negative Cheetah voltage makes positive GPT integrated voltage
# accelerate at the same nominal phase sign. Verify with a GPT phase scan.
CAVITY_VOLTAGE_SIGN = -1.0
CAVITY_PHASE_OFFSET_DEG = 0.0

# GPT survey centres [m] measured from IH-DTL1 exit.
CENTERS = {
    "S1": 0.200,
    "S2": 0.280,
    "Q1": 0.410,
    "Q2": 0.570,
    "BC1": 0.755,
    "Q3": 0.930,
    "Q4": 1.090,
    "DIAG1": 1.390,
    "BC2": 1.675,
    "Q5": 1.850,
    "Q6": 2.010,
    "BM1": 2.205,
    "TQ1": 2.560,
    "TQ2": 2.910,
    "TQ3": 3.260,
    "BM2": 3.535,
    "BC3": 3.755,
    "Q7": 3.940,
    "Q8": 4.120,
    "Q9": 4.300,
    "Q10": 4.480,
    "BC4": 4.675,
    "DIAG2": 4.900,
    "EXIT": 5.185,
}

REF_PLANES = {
    "REF1": 0.345,
    "REF2": 0.855,
    "REF3": 1.775,
    "REF4": 3.855,
    "REF5": 4.775,
}

GRADIENT_T_PER_M = {
    "Q1": 2.57886,
    "Q2": -2.96410,
    "Q3": 2.6200,
    "Q4": -2.3100,
    "Q5": 2.4500,
    "Q6": -2.3000,
    "TQ1": 4.1060,
    "TQ2": -3.8000,
    "TQ3": 4.1060,
    "Q7": -1.61976,
    "Q8": 5.60973,
    "Q9": -5.36216,
    "Q10": 2.34460,
}

STEERER_FIELD_T = {
    "S1": 0.05655885965272645,
    "S2": -0.03736744519301249,
}

CAVITY_VOLTAGE_KV = {
    "BC1": 47.03,
    "BC2": 64.67,
    "BC3": 75.23,
    "BC4": 111.39,
}

# v4 default phases are intentionally placeholders pending re-matching.
CAVITY_PHASE_GPT_DEG = {
    "BC1": 0.0,
    "BC2": 0.0,
    "BC3": 0.0,
    "BC4": 0.0,
}


def gradient_to_k1(gradient_t_per_m: float) -> float:
    """Physical quadrupole gradient -> Cheetah normalized k1."""
    return 0.299792458 * gradient_t_per_m / DESIGN_P0C_GEV


def field_to_corrector_angle(field_t: float, length_m: float) -> float:
    """Uniform transverse B field -> thin-kick angle at design rigidity."""
    return field_t * length_m / BRHO_TM


def element_metadata(**kwargs):
    out = {
        "source": "GPT mebt2_v4_324MHz.in",
        "design_kinetic_energy_MeV": DESIGN_KINETIC_ENERGY_MEV,
    }
    out.update(kwargs)
    return out


def make_lattice_dict() -> dict:
    elements: dict[str, list] = {}
    sec1: list[str] = []
    sec2: list[str] = []
    sec3: list[str] = []

    def add(section: list[str], name: str, cls: str, params: dict):
        if name in elements:
            raise ValueError(f"Duplicate element name: {name}")
        elements[name] = [cls, params]
        section.append(name)

    def drift(section: list[str], name: str, length: float):
        if length < -1e-12:
            raise ValueError(f"Negative drift {name}: {length}")
        add(
            section,
            name,
            "Drift",
            {
                "tracking_method": "linear",
                "length": float(max(length, 0.0)),
                "metadata": element_metadata(kind="survey_drift"),
            },
        )

    def marker(section: list[str], name: str, s_m: float):
        add(
            section,
            name,
            "Marker",
            {"metadata": element_metadata(kind="marker", gpt_s_m=float(s_m))},
        )

    def quad(section: list[str], name: str):
        grad = GRADIENT_T_PER_M[name]
        add(
            section,
            name,
            "Quadrupole",
            {
                "tracking_method": "linear",
                "length": QUAD_LENGTH_M,
                "k1": gradient_to_k1(grad),
                "misalignment": [0.0, 0.0],
                "tilt": 0.0,
                "num_steps": 1,
                "metadata": element_metadata(
                    kind="quadrupole",
                    gpt_center_s_m=CENTERS[name],
                    gpt_gradient_T_per_m=grad,
                    k1_reference_p0c_MeV=DESIGN_P0C_MEV,
                ),
            },
        )

    def cavity(section: list[str], name: str):
        vgpt = CAVITY_VOLTAGE_KV[name] * 1e3
        phigpt = CAVITY_PHASE_GPT_DEG[name]
        add(
            section,
            name,
            "Cavity",
            {
                "length": CAVITY_EFFECTIVE_LENGTH_M,
                "voltage": CAVITY_VOLTAGE_SIGN * vgpt,
                "phase": phigpt + CAVITY_PHASE_OFFSET_DEG,
                "frequency": RF_FREQUENCY_HZ,
                "cavity_type": "standing_wave",
                "metadata": element_metadata(
                    kind="cavity",
                    gpt_center_s_m=CENTERS[name],
                    gpt_integrated_voltage_V=vgpt,
                    gpt_phase_deg=phigpt,
                    gpt_mechanical_length_m=CAVITY_MECHANICAL_LENGTH_M,
                    gpt_effective_length_m=CAVITY_EFFECTIVE_LENGTH_M,
                    cheetah_voltage_sign_adapter=CAVITY_VOLTAGE_SIGN,
                    cheetah_phase_offset_deg=CAVITY_PHASE_OFFSET_DEG,
                ),
            },
        )

    # ------------------------------------------------------------------
    # SECTION 1: IH-DTL1 exit -> BM1 entrance
    # ------------------------------------------------------------------
    marker(sec1, "MEBT2_ENT", 0.0)
    drift(sec1, "D_ENT_IH_COVER_EXIT", 0.055)
    marker(sec1, "IH_COVER_EXIT", 0.055)

    drift(sec1, "D_IH_COVER_S1", 0.130)
    add(
        sec1,
        "S1",
        "VerticalCorrector",
        {
            "length": CORRECTOR_LENGTH_M,
            "angle": field_to_corrector_angle(
                STEERER_FIELD_T["S1"], CORRECTOR_LENGTH_M
            ),
            "metadata": element_metadata(
                kind="vertical_corrector",
                gpt_center_s_m=CENTERS["S1"],
                gpt_Bx_T=STEERER_FIELD_T["S1"],
                angle_reference_p0c_MeV=DESIGN_P0C_MEV,
            ),
        },
    )

    drift(sec1, "D_S1_S2", 0.050)
    add(
        sec1,
        "S2",
        "VerticalCorrector",
        {
            "length": CORRECTOR_LENGTH_M,
            "angle": field_to_corrector_angle(
                STEERER_FIELD_T["S2"], CORRECTOR_LENGTH_M
            ),
            "metadata": element_metadata(
                kind="vertical_corrector",
                gpt_center_s_m=CENTERS["S2"],
                gpt_Bx_T=STEERER_FIELD_T["S2"],
                angle_reference_p0c_MeV=DESIGN_P0C_MEV,
            ),
        },
    )

    drift(sec1, "D_S2_REF1", 0.050)
    marker(sec1, "REF1", REF_PLANES["REF1"])
    drift(sec1, "D_REF1_Q1", 0.005)
    quad(sec1, "Q1")
    drift(sec1, "D_Q1_Q2", 0.040)
    quad(sec1, "Q2")

    drift(sec1, "D_Q2_BC1_MECH_IN", 0.088)
    drift(sec1, "BC1_PRE", CAVITY_SIDE_DRIFT_M)
    cavity(sec1, "BC1")
    drift(sec1, "BC1_POST", CAVITY_SIDE_DRIFT_M)

    drift(sec1, "D_BC1_REF2", 0.063)
    marker(sec1, "REF2", REF_PLANES["REF2"])
    drift(sec1, "D_REF2_Q3", 0.015)
    quad(sec1, "Q3")
    drift(sec1, "D_Q3_Q4", 0.040)
    quad(sec1, "Q4")

    drift(sec1, "D_Q4_DIAG1", 0.240)
    marker(sec1, "DIAG1", CENTERS["DIAG1"])

    drift(sec1, "D_DIAG1_BC2_MECH_IN", 0.248)
    drift(sec1, "BC2_PRE", CAVITY_SIDE_DRIFT_M)
    cavity(sec1, "BC2")
    drift(sec1, "BC2_POST", CAVITY_SIDE_DRIFT_M)

    drift(sec1, "D_BC2_REF3", 0.063)
    marker(sec1, "REF3", REF_PLANES["REF3"])
    drift(sec1, "D_REF3_Q5", 0.015)
    quad(sec1, "Q5")
    drift(sec1, "D_Q5_Q6", 0.040)
    quad(sec1, "Q6")

    bm1_in = CENTERS["BM1"] - BEND_LENGTH_M / 2.0
    drift(sec1, "D_Q6_BM1_IN", bm1_in - 2.070)
    marker(sec1, "BM1_IN", bm1_in)

    # ------------------------------------------------------------------
    # SECTION 2: BM1 -> triplet -> BM2
    # ------------------------------------------------------------------
    add(
        sec2,
        "BM1",
        "Dipole",
        {
            "tracking_method": "linear",
            "length": BEND_LENGTH_M,
            "angle": BEND_ANGLE_RAD,
            "k1": 0.0,
            "dipole_e1": 0.0,
            "dipole_e2": 0.0,
            "tilt": 0.0,
            "gap": BEND_GAP_M,
            "gap_exit": BEND_GAP_M,
            "fringe_integral": 0.0,
            "fringe_integral_exit": 0.0,
            "fringe_at": "both",
            "fringe_type": "linear_edge",
            "metadata": element_metadata(
                kind="sector_dipole",
                gpt_center_s_m=CENTERS["BM1"],
                radius_m=BEND_RADIUS_M,
                bend_angle_deg=BEND_ANGLE_DEG,
                gpt_full_gap_m=BEND_GAP_M,
            ),
        },
    )

    bm1_out = CENTERS["BM1"] + BEND_LENGTH_M / 2.0
    drift(sec2, "D_BM1_TQ1", (CENTERS["TQ1"] - QUAD_LENGTH_M / 2) - bm1_out)
    quad(sec2, "TQ1")
    drift(sec2, "D_TQ1_TQ2", 0.230)
    quad(sec2, "TQ2")
    drift(sec2, "D_TQ2_TQ3", 0.230)
    quad(sec2, "TQ3")

    bm2_in = CENTERS["BM2"] - BEND_LENGTH_M / 2.0
    drift(sec2, "D_TQ3_BM2", bm2_in - (CENTERS["TQ3"] + QUAD_LENGTH_M / 2.0))

    add(
        sec2,
        "BM2",
        "Dipole",
        {
            "tracking_method": "linear",
            "length": BEND_LENGTH_M,
            "angle": BEND_ANGLE_RAD,
            "k1": 0.0,
            "dipole_e1": 0.0,
            "dipole_e2": 0.0,
            "tilt": 0.0,
            "gap": BEND_GAP_M,
            "gap_exit": BEND_GAP_M,
            "fringe_integral": 0.0,
            "fringe_integral_exit": 0.0,
            "fringe_at": "both",
            "fringe_type": "linear_edge",
            "metadata": element_metadata(
                kind="sector_dipole",
                gpt_center_s_m=CENTERS["BM2"],
                radius_m=BEND_RADIUS_M,
                bend_angle_deg=BEND_ANGLE_DEG,
                gpt_full_gap_m=BEND_GAP_M,
            ),
        },
    )
    bm2_out = CENTERS["BM2"] + BEND_LENGTH_M / 2.0
    marker(sec2, "BM2_OUT", bm2_out)

    # ------------------------------------------------------------------
    # SECTION 3: BM2 exit -> MEBT-II exit
    # ------------------------------------------------------------------
    bc3_mech_in = CENTERS["BC3"] - CAVITY_MECHANICAL_LENGTH_M / 2.0
    drift(sec3, "D_BM2_BC3_MECH_IN", bc3_mech_in - bm2_out)
    drift(sec3, "BC3_PRE", CAVITY_SIDE_DRIFT_M)
    cavity(sec3, "BC3")
    drift(sec3, "BC3_POST", CAVITY_SIDE_DRIFT_M)

    drift(sec3, "D_BC3_REF4", REF_PLANES["REF4"] - (
        CENTERS["BC3"] + CAVITY_MECHANICAL_LENGTH_M / 2.0
    ))
    marker(sec3, "REF4", REF_PLANES["REF4"])
    drift(sec3, "D_REF4_Q7", (CENTERS["Q7"] - QUAD_LENGTH_M / 2.0) - REF_PLANES["REF4"])
    quad(sec3, "Q7")
    drift(sec3, "D_Q7_Q8", 0.060)
    quad(sec3, "Q8")
    drift(sec3, "D_Q8_Q9", 0.060)
    quad(sec3, "Q9")
    drift(sec3, "D_Q9_Q10", 0.060)
    quad(sec3, "Q10")

    bc4_mech_in = CENTERS["BC4"] - CAVITY_MECHANICAL_LENGTH_M / 2.0
    q10_out = CENTERS["Q10"] + QUAD_LENGTH_M / 2.0
    drift(sec3, "D_Q10_BC4_MECH_IN", bc4_mech_in - q10_out)
    drift(sec3, "BC4_PRE", CAVITY_SIDE_DRIFT_M)
    cavity(sec3, "BC4")
    drift(sec3, "BC4_POST", CAVITY_SIDE_DRIFT_M)

    bc4_mech_out = CENTERS["BC4"] + CAVITY_MECHANICAL_LENGTH_M / 2.0
    drift(sec3, "D_BC4_REF5", REF_PLANES["REF5"] - bc4_mech_out)
    marker(sec3, "REF5", REF_PLANES["REF5"])
    drift(sec3, "D_REF5_DIAG2", CENTERS["DIAG2"] - REF_PLANES["REF5"])
    marker(sec3, "DIAG2", CENTERS["DIAG2"])
    drift(sec3, "D_DIAG2_EXIT", CENTERS["EXIT"] - CENTERS["DIAG2"])
    marker(sec3, "MEBT2_EXIT", CENTERS["EXIT"])

    return {
        "version": "cheetah-0.8",
        "title": "J-PARC muon MEBT-II 324 MHz validation lattice",
        "info": (
            "Cheetah adaptation of GPT mebt2_v4_324MHz.in. "
            "Three nested sections: upstream, achromat, downstream. "
            "Physical GPT settings retained in element metadata."
        ),
        "root": "MEBT2",
        "elements": elements,
        "lattices": {
            "SEC1_UPSTREAM": sec1,
            "SEC2_ACHROMAT": sec2,
            "SEC3_DOWNSTREAM": sec3,
            "MEBT2": ["SEC1_UPSTREAM", "SEC2_ACHROMAT", "SEC3_DOWNSTREAM"],
        },
    }


def sequence_length(lattice_dict: dict, lattice_name: str) -> float:
    """Recursively compute LatticeJSON path length."""
    elements = lattice_dict["elements"]
    lattices = lattice_dict["lattices"]
    total = 0.0
    for name in lattices[lattice_name]:
        if name in lattices:
            total += sequence_length(lattice_dict, name)
        else:
            total += float(elements[name][1].get("length", 0.0))
    return total


def main(output: str | Path = "mebt2_lattice.json") -> Path:
    output = Path(output)
    lattice = make_lattice_dict()
    length = sequence_length(lattice, "MEBT2")
    if not math.isclose(length, CENTERS["EXIT"], abs_tol=2e-12):
        raise RuntimeError(
            f"Lattice length mismatch: {length:.12f} m != {CENTERS['EXIT']:.12f} m"
        )
    output.write_text(json.dumps(lattice, indent=4) + "\n")
    print(f"Wrote {output}")
    print(f"Total MEBT-II path length = {length:.9f} m")
    print(f"Design kinetic energy = {DESIGN_KINETIC_ENERGY_MEV:.9f} MeV")
    print(f"Design p0c = {DESIGN_P0C_MEV:.9f} MeV/c")
    print(f"B rho = {BRHO_TM:.9f} T m")
    return output


if __name__ == "__main__":
    main()
