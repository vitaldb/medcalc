# MedCalc MCP Server

A comprehensive medical calculation MCP (Model Context Protocol) server providing 59 clinical tools for healthcare professionals and AI assistants.

## Installation

```bash
pip install medcalc
```

## Usage

Start the MCP server:

```bash
medcalc
```

## Claude Desktop / `uvx` Integration

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "medcalc": {
      "command": "uvx",
      "args": ["medcalc@latest"]
    }
  }
}
```

## Available Tools (59)

### Renal Function
| Tool | Description |
|------|-------------|
| `egfr_epi` | eGFR using 2021 CKD-EPI Creatinine equation |
| `egfr_epi_cr_cys` | eGFR using 2021 CKD-EPI Creatinine-Cystatin C equation |
| `mdrd_gfr` | GFR using MDRD equation |
| `crcl_cockcroft_gault` | Creatinine Clearance (Cockcroft-Gault) |
| `fractional_excretion_of_sodium` | FENa for AKI evaluation |
| `free_water_deficit` | Free water deficit for hypernatremia |

### Electrolytes & ABG
| Tool | Description |
|------|-------------|
| `corrected_calcium` | Calcium corrected for albumin |
| `corrected_sodium_katz` | Corrected sodium (Katz formula) |
| `corrected_sodium_hillier` | Corrected sodium (Hillier formula) |
| `serum_anion_gap` | Serum anion gap with delta-delta |
| `serum_osmolarity` | Serum osmolarity (calculated) |
| `anion_gap_delta_delta` | Delta-delta ratio for mixed acid-base disorders |
| `aa_gradient` | A-a gradient |
| `winters_formula` | Winter's formula for expected pCO2 |

### Anthropometrics & Fluids
| Tool | Description |
|------|-------------|
| `bmi_calculator` | Body Mass Index |
| `bsa_calculator` | Body Surface Area (Mosteller) |
| `ibw_calculator` | Ideal Body Weight (Devine) |
| `abw_calculator` | Adjusted Body Weight |
| `maintenance_fluids` | IV maintenance fluids (4-2-1 rule) |
| `ventilator_tidal_volume` | IBW-based tidal volume (6-8 mL/kg) |

### Cardiovascular Risk
| Tool | Description |
|------|-------------|
| `framingham_risk_score` | 10-year CHD risk (Framingham) |
| `prevent_cvd_risk` | AHA PREVENT 10-year CVD risk |
| `ascvd_10yr_risk` | ACC/AHA Pooled Cohort ASCVD risk |
| `revised_cardiac_risk_index` | RCRI for pre-operative cardiac risk |
| `heart_score` | HEART score for chest pain evaluation |
| `duke_activity_status_index` | DASI functional capacity (METs) |
| `gupta_perioperative_mica` | Gupta perioperative MI/cardiac arrest risk |

### Cardiac & Vascular
| Tool | Description |
|------|-------------|
| `map_calculator` | Mean Arterial Pressure |
| `qtc_calculator` | Corrected QT Interval (Bazett, Fridericia, etc.) |
| `chads2_vasc_score` | CHA₂DS₂-VASc stroke risk |
| `has_bled_score` | HAS-BLED bleeding risk |
| `grace_acs_score` | GRACE ACS mortality risk |
| `timi_stemi` | TIMI score for STEMI |
| `timi_nstemi` | TIMI score for NSTEMI/UA |

### Pulmonary & VTE
| Tool | Description |
|------|-------------|
| `wells_pe_criteria` | Wells' criteria for PE |
| `wells_dvt_criteria` | Wells' criteria for DVT |
| `perc_rule` | PERC rule for PE exclusion |
| `ariscat_score` | ARISCAT postoperative pulmonary complications |
| `curb65_score` | CURB-65 for pneumonia severity |
| `psi_port_score` | PSI/PORT score for CAP |
| `stop_bang_score` | STOP-BANG for obstructive sleep apnea |

### Hepatology
| Tool | Description |
|------|-------------|
| `child_pugh_score` | Child-Pugh cirrhosis classification |
| `meld_3` | MELD 3.0 score |
| `fib4_index` | FIB-4 index for liver fibrosis |

### Critical Care
| Tool | Description |
|------|-------------|
| `sofa_score` | Sequential Organ Failure Assessment |
| `apache2_score` | APACHE II ICU severity |
| `sepsis_criteria` | qSOFA / SIRS / Sepsis-3 criteria |
| `glasgow_coma_scale` | Glasgow Coma Scale |
| `news2_score` | National Early Warning Score 2 |

### Neurology & Psychiatry
| Tool | Description |
|------|-------------|
| `nihss_score` | NIH Stroke Scale |
| `phq9_score` | PHQ-9 depression screening |
| `gad7_score` | GAD-7 anxiety screening |
| `ciwa_ar_score` | CIWA-Ar alcohol withdrawal |

### Pediatrics
| Tool | Description |
|------|-------------|
| `bp_children` | Pediatric blood pressure percentiles |
| `pecarn_pediatric_head_injury` | PECARN pediatric head CT decision rule |

### Hematology & Surgery
| Tool | Description |
|------|-------------|
| `caprini_score` | Caprini VTE risk in surgical patients |
| `centor_score_modified` | Modified Centor (McIsaac) for pharyngitis |
| `ldl_calculated` | LDL cholesterol (Friedewald) |
| `homa_ir` | HOMA-IR insulin resistance |

### Pharmacology
| Tool | Description |
|------|-------------|
| `steroid_conversion` | Corticosteroid dose conversion |
| `calculate_mme` | Morphine Milligram Equivalents |
| `infusion_rate` | Universal IV drug infusion rate calculator (mL/h) |

### Obstetrics
| Tool | Description |
|------|-------------|
| `pregnancy_calculator` | Pregnancy due date calculations |

## License

MIT
