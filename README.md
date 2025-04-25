# Medical Calculation Server
[![smithery badge](https://smithery.ai/badge/@vitaldb/medcalc)](https://smithery.ai/server/@vitaldb/medcalc)

Based on VitalDB APIs, this code performs various medical calculations.

## Supported Formula
TODO
## Usage

### Installing via Smithery

To install Medical Calculation Server for Claude Desktop automatically via [Smithery](https://smithery.ai/server/@vitaldb/medcalc):

```bash
npx -y @smithery/cli install @vitaldb/medcalc --client claude
```

### Installing in Node.js
```sh
npm install --save @vitaldb/medcalc
```
For ES module:
```sh
import { vitalCapacityBW, ... } from '@vitaldb/medcalc';
```
For CommonJS:
```sh
const { vitalCapacityBW, ... } = require('@vitaldb/medcalc');
```

### As MCP Server
Run with Node.js
```sh
npx @vitaldb/medcalc serve
```

Example: MCP server for GlamaAI that allows to retrieve a resourse list
```
- init {[resources: true]}
  - → {result: {capabilities: ["resources"], version: "0.0.5"}}
- resources/list {}
  - → {result: [{"uri":"formula://Vital Capacity by Body Weight","name":"Vital Capacity","type":"formula"}]}
```

### Examples
For example, calculate VC (vital capacity) (ml) by body weight with 70 kg
```js
import { vitalCapacityBW } from '@vitaldb/medcalc';
console.log(vitalCapacityBW(70)); // 4200
```
