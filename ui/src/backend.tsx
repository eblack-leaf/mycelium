import { invoke } from "@tauri-apps/api/core";
import { createStore, reconcile, SetStoreFunction, Store } from "solid-js/store";
import { Block } from "./bindings/Block.ts";
import { PasteResult } from "./bindings/PasteResult.ts";
import { PlaceholderValue } from "./bindings/PlaceholderValue.ts";
import { Settings } from "./bindings/Settings.ts";
import { Suggestions } from "./bindings/Suggestions.ts";

export class Backend {
    blocks: [Store<Block[]>, SetStoreFunction<Block[]>];
    suggestions: [Store<Suggestions>, SetStoreFunction<Suggestions>];
    values: [Store<PlaceholderValue[]>, SetStoreFunction<PlaceholderValue[]>];
    settings: [Store<Settings>, SetStoreFunction<Settings>];

    constructor() {
        this.blocks = createStore<Block[]>([]);
        this.suggestions = createStore<Suggestions>({ placeholders: [], schema: [], other: [] });
        this.values = createStore<PlaceholderValue[]>([]);
        this.settings = createStore<Settings>({
            surreal_endpoint:  "ws://localhost:8000",
            surreal_namespace: "test",
            surreal_database:  "test",
            surreal_username:  "root",
            surreal_password:  "root",
            placeholder_prefix: "@",
        });
    }

    composingBlock(): Block | undefined {
        const all = this.blocks[0];
        return all[all.length - 1]?.state === "Composing"
            ? all[all.length - 1]
            : undefined;
    }

    async init(): Promise<void> {
        const [blocks, sugs, vals, cfg] = await Promise.all([
            invoke<Block[]>("blocks"),
            invoke<Suggestions>("suggestions"),
            invoke<PlaceholderValue[]>("get_values"),
            invoke<Settings>("get_settings"),
        ]);
        this.blocks[1](reconcile(blocks));
        this.suggestions[1](reconcile(sugs));
        this.values[1](reconcile(vals));
        this.settings[1](reconcile(cfg));
        // Attempt schema refresh in background — fails silently if DB not reachable
        this.refreshSchema();
    }

    async submitBlock(id: string, query: string): Promise<void> {
        const blocks = await invoke<Block[]>("submit_block", { id, query });
        this.blocks[1](reconcile(blocks));
        // Refresh suggestions so placeholders stay in sync
        const sugs = await invoke<Suggestions>("suggestions");
        this.suggestions[1](reconcile(sugs));
    }

    async saveValue(name: string, value: string): Promise<void> {
        const vals = await invoke<PlaceholderValue[]>("save_value", { name, value });
        this.values[1](reconcile(vals));
        const sugs = await invoke<Suggestions>("suggestions");
        this.suggestions[1](reconcile(sugs));
    }

    async deleteValue(name: string): Promise<void> {
        const vals = await invoke<PlaceholderValue[]>("delete_value", { name });
        this.values[1](reconcile(vals));
        const sugs = await invoke<Suggestions>("suggestions");
        this.suggestions[1](reconcile(sugs));
    }

    async renameValue(oldName: string, newName: string): Promise<void> {
        const vals = await invoke<PlaceholderValue[]>("rename_value", {
            oldName,
            newName,
        });
        this.values[1](reconcile(vals));
        const sugs = await invoke<Suggestions>("suggestions");
        this.suggestions[1](reconcile(sugs));
    }

    async updateSettings(patch: Partial<Settings>): Promise<void> {
        const current = this.settings[0];
        const next: Settings = { ...current, ...patch };
        const cfg = await invoke<Settings>("update_settings", { settings: next });
        this.settings[1](reconcile(cfg));
    }

    async suggestName(context: string): Promise<string> {
        return invoke<string>("suggest_name", { context });
    }

    async filterSuggestions(word: string): Promise<void> {
        const sugs = await invoke<Suggestions>("filter_suggestions", { word });
        this.suggestions[1](reconcile(sugs));
    }

    async refreshSchema(): Promise<string | null> {
        try {
            await invoke("refresh_schema");
            const sugs = await invoke<Suggestions>("suggestions");
            this.suggestions[1](reconcile(sugs));
            return null;
        } catch (e) {
            return String(e);
        }
    }

    async pasteValue(context: string, value: string): Promise<string> {
        const result = await invoke<PasteResult>("paste_value", { context, value });
        this.values[1](reconcile(result.values));
        const sugs = await invoke<Suggestions>("suggestions");
        this.suggestions[1](reconcile(sugs));
        return result.name;
    }
}
