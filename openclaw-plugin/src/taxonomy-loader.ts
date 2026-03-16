import { readFileSync, existsSync } from "node:fs";
import { resolve } from "node:path";
import { load as yamlLoad } from "js-yaml";

export type TaxonomyEntry = {
  id: string;
  name: string;
  trigger: string;
  fix_sop: string;
  prevention: string;
  confidence: number;
  source_exps: string[];
  keywords: string[];
  created_at: string;
  last_verified: string;
};

export type ErrorTaxonomy = {
  taxonomy: Record<string, Record<string, TaxonomyEntry[]>>;
};

export function loadTaxonomy(workspaceRoot: string, taxonomyPath: string): ErrorTaxonomy | null {
  const fullPath = resolve(workspaceRoot, taxonomyPath);
  if (!existsSync(fullPath)) return null;

  try {
    const raw = yamlLoad(readFileSync(fullPath, "utf-8")) as Record<string, unknown>;
    if (!raw?.taxonomy || typeof raw.taxonomy !== "object") return null;
    return raw as unknown as ErrorTaxonomy;
  } catch {
    return null;
  }
}

export function buildKeywordIndex(taxonomy: ErrorTaxonomy): Map<string, TaxonomyEntry[]> {
  const index = new Map<string, TaxonomyEntry[]>();

  for (const domain of Object.values(taxonomy.taxonomy)) {
    for (const entries of Object.values(domain)) {
      for (const entry of entries) {
        for (const kw of entry.keywords) {
          const lower = kw.toLowerCase();
          const existing = index.get(lower);
          if (existing) {
            existing.push(entry);
          } else {
            index.set(lower, [entry]);
          }
        }
      }
    }
  }

  return index;
}
