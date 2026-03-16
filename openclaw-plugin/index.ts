/**
 * OpenClaw Meta-Learning Plugin
 *
 * Layer 1 of the three-layer self-evolving learning architecture:
 * - before_prompt_build: Quick Think risk assessment (< 50ms, no LLM)
 * - agent_end: Signal Capture for post-task learning signals
 *
 * Layer 2/3 are triggered via Python CLI (heartbeat or cron).
 */

import type { OpenClawPluginApi } from "openclaw/plugin-sdk";
import { loadTaxonomy } from "./src/taxonomy-loader.js";
import { quickThinkEvaluate, formatRiskWarning } from "./src/quick-think.js";
import { analyzeMessagesForSignal, writeSignal } from "./src/signal-writer.js";

type MetaLearningConfig = {
  workspaceRoot?: string;
  signalBufferDir?: string;
  taxonomyPath?: string;
  pythonBin?: string;
  enabled?: boolean;
};

const DEFAULTS = {
  signalBufferDir: "signal_buffer",
  taxonomyPath: "error_taxonomy.yaml",
  pythonBin: "venv/bin/python3",
};

const plugin = {
  id: "meta-learning",
  name: "Meta-Learning",
  description: "Three-layer self-evolving learning system for agent improvement",

  register(api: OpenClawPluginApi) {
    const cfg = (api.pluginConfig ?? {}) as MetaLearningConfig;
    if (cfg.enabled === false) {
      api.logger.info("meta-learning: disabled by config");
      return;
    }

    const signalBufferDir = cfg.signalBufferDir ?? DEFAULTS.signalBufferDir;
    const taxonomyPath = cfg.taxonomyPath ?? DEFAULTS.taxonomyPath;

    const resolveWorkspace = (): string => {
      return cfg.workspaceRoot ?? api.resolvePath(".");
    };

    // ========================================================================
    // Quick Think: inject risk assessment before agent processes the message
    // ========================================================================
    api.on("before_prompt_build", async (event) => {
      const workspaceRoot = resolveWorkspace();

      try {
        const taxonomy = loadTaxonomy(workspaceRoot, taxonomyPath);
        const userMessage = extractLastUserMessage(event.messages);
        if (!userMessage) return;

        const result = quickThinkEvaluate(taxonomy, userMessage);
        if (!result.hit) return;

        api.logger.info?.(
          `meta-learning: QuickThink hit — risk=${result.riskLevel}, signals=${result.matchedSignals.join(",")}`,
        );

        return {
          prependContext: formatRiskWarning(result, taxonomy),
        };
      } catch (err) {
        api.logger.warn(`meta-learning: QuickThink error: ${String(err)}`);
      }
    });

    // ========================================================================
    // Signal Capture: analyze completed conversation for learning signals
    // ========================================================================
    api.on("agent_end", async (event, ctx) => {
      if (!event.success || !event.messages || event.messages.length === 0) return;

      const workspaceRoot = resolveWorkspace();

      try {
        const signal = analyzeMessagesForSignal(event.messages, ctx.sessionId);
        if (!signal) return;

        const filePath = writeSignal(workspaceRoot, signalBufferDir, signal);
        api.logger.info?.(
          `meta-learning: Signal captured [${signal.trigger_reason}] → ${filePath}`,
        );
      } catch (err) {
        api.logger.warn(`meta-learning: Signal capture error: ${String(err)}`);
      }
    });

    // ========================================================================
    // Tool tracking: count tool calls for efficiency anomaly detection
    // ========================================================================
    let toolCallCount = 0;

    api.on("before_tool_call", async () => {
      toolCallCount++;
    });

    api.on("session_start", async () => {
      toolCallCount = 0;
    });

    // ========================================================================
    // Service registration
    // ========================================================================
    api.registerService({
      id: "meta-learning",
      start: () => {
        api.logger.info(
          `meta-learning: initialized (taxonomy=${taxonomyPath}, signalBuffer=${signalBufferDir})`,
        );
      },
      stop: () => {
        api.logger.info("meta-learning: stopped");
      },
    });
  },
};

function extractLastUserMessage(messages: unknown[]): string | null {
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i] as Record<string, unknown> | null;
    if (!msg || msg.role !== "user") continue;

    if (typeof msg.content === "string") return msg.content;
    if (Array.isArray(msg.content)) {
      const texts = (msg.content as Array<Record<string, unknown>>)
        .filter((b) => b?.type === "text" && typeof b.text === "string")
        .map((b) => b.text as string);
      if (texts.length > 0) return texts.join(" ");
    }
  }
  return null;
}

export default plugin;
