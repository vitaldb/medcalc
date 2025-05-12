# MedCalc MCP Server

This package provides a set of medical calculation tools exposed as an MCP (Model Context Protocol) server.

## Installation

```bash
pip install medcalc
```

This command installs the package and makes the `medcalc` command available in your environment.

## Usage

The primary way to use this package is to run it as an MCP server. Once installed, you can start the server by running the following command in your terminal:

This will start the FastMCP server, making the following medical calculation tools available for MCP clients:

*   `egfr_epi`: Calculates eGFR using the 2021 CKD-EPI Creatinine equation.
*   `egfr_epi_cr_cys`: Calculates eGFR using the 2021 CKD-EPI Creatinine-Cystatin C equation.
*   `bp_children`: Calculates pediatric blood pressure percentiles.
*   `bmi_bsa_calculator`: Calculates Body Mass Index (BMI) and Body Surface Area (BSA).
*   `crcl_cockcroft_gault`: Calculates Creatinine Clearance using Cockcroft-Gault.
*   `map_calculator`: Calculates Mean Arterial Pressure (MAP).
*   `chads2_vasc_score`: Calculates CHA₂DS₂-VASc score for stroke risk.
*   `prevent_cvd_risk`: Predicts 10-year risk of Cardiovascular Disease Events (PREVENT).
*   `corrected_calcium`: Calculates corrected calcium for albumin levels.
*   `qtc_calculator`: Calculates Corrected QT Interval (QTc) using various formulas.
*   `wells_pe_criteria`: Calculates Wells' Criteria score for Pulmonary Embolism risk.
*   `ibw_abw_calculator`: Calculates Ideal Body Weight (IBW) and Adjusted Body Weight (ABW).
*   `pregnancy_calculator`: Calculates pregnancy due dates.
*   `revised_cardiac_risk_index`: Calculates RCRI for pre-operative cardiac risk.
*   `child_pugh_score`: Calculates Child-Pugh score for cirrhosis.
*   `steroid_conversion`: Converts corticosteroid dosages.
*   `calculate_mme`: Calculates Morphine Milligram Equivalents (MME).
*   `maintenance_fluids`: Calculates maintenance IV fluid rate (4-2-1 Rule).
*   `corrected_sodium`: Calculates corrected sodium for hyperglycemia.
*   `meld_3`: Calculates MELD 3.0 score for liver disease.
*   `framingham_risk_score`: Calculates Framingham Risk Score for 10-year CHD risk.
*   `homa_ir`: Calculates HOMA-IR score for insulin resistance.

## Claude Desktop / `uvx` Integration

This server can be integrated with applications like Claude Desktop that support MCP. To configure Claude Desktop (or another FastAgent application) to use this server, you can add an entry to your `claude_desktop_config.json`.

Assuming you have `uvx` installed and configured, you can typically run an installed Python package command via `uvx`. Here's an example configuration:

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

**Note:** Ensure `uvx` is installed (`pip install uvx`) and the `medcalc` package is installed in an environment accessible by `uvx`. The exact `uvx` command might vary based on your specific `uvx` setup and environment management.
