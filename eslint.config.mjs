import { defineConfig, globalIgnores } from "eslint/config";
import nextVitals from "eslint-config-next/core-web-vitals";
import nextTs from "eslint-config-next/typescript";

const eslintConfig = defineConfig([
  ...nextVitals,
  ...nextTs,
  {
    // Golden rule 4: src/core is pure TypeScript. It must stay testable in
    // isolation, so it cannot reach into the UI, the stores or the server.
    files: ["src/core/**/*.ts"],
    rules: {
      "no-restricted-imports": [
        "error",
        {
          paths: ["react", "react-dom", "three", "zustand", "motion"],
          patterns: [
            "next",
            "next/*",
            "react/*",
            "@react-three/*",
            "@/app/*",
            "@/features/*",
            "@/components/*",
            "@/stores/*",
            "@/server/*",
            "node:*",
          ],
        },
      ],
      "no-restricted-globals": ["error", "window", "document", "localStorage", "fetch"],
    },
  },
  {
    // Tests and the Node-only test helper may read fixture files.
    files: ["src/core/**/*.test.ts", "src/core/**/test-utils.ts"],
    rules: { "no-restricted-imports": "off" },
  },
  globalIgnores([".next/**", "out/**", "build/**", "next-env.d.ts", ".cache/**"]),
]);

export default eslintConfig;
