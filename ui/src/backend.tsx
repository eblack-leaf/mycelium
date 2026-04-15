import { invoke } from "@tauri-apps/api/core";
import { createStore, reconcile, SetStoreFunction, Store } from "solid-js/store";
import { Block } from "./bindings/Block.ts";
import { PasteResult } from "./bindings/PasteResult.ts";
import { PlaceholderValue } from "./bindings/PlaceholderValue.ts";
import { Settings } from "./bindings/Settings.ts";
import { Suggestions } from "./bindings/Suggestions.ts";
import { TaskMeta } from "./bindings/TaskMeta.ts";

export class Backend {
    blocks: [Store<Block[]>, SetStoreFunction<Block[]>];
    suggestions: [Store<Suggestions>, SetStoreFunction<Suggestions>];
    values: [Store<PlaceholderValue[]>, SetStoreFunction<PlaceholderValue[]>];
    settings: [Store<Settings>, SetStoreFunction<Settings>];
    tasks: [Store<TaskMeta[]>, SetStoreFunction<TaskMeta[]>];

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
        this.tasks = createStore<TaskMeta[]>([]);
    }

    composingBlock(): Block | undefined {
        const all = this.blocks[0];
        return all[all.length - 1]?.state === "Composing"
            ? all[all.length - 1]
            : undefined;
    }

    async init(): Promise<void> {
        const [blocks, sugs, vals, cfg, taskList] = await Promise.all([
            invoke<Block[]>("blocks"),
            invoke<Suggestions>("suggestions"),
            invoke<PlaceholderValue[]>("get_values"),
            invoke<Settings>("get_settings"),
            invoke<TaskMeta[]>("list_tasks"),
        ]);
        this.blocks[1](reconcile(blocks));
        this.suggestions[1](reconcile(sugs));
        this.values[1](reconcile(vals));
        this.settings[1](reconcile(cfg));
        this.tasks[1](reconcile(taskList));
        // Attempt schema refresh in background — fails silently if DB not reachable
        this.refreshSchema();
    }

    async submitBlock(id: string, query: string): Promise<void> {
        const blocks = await invoke<Block[]>("submit_block", { id, query });
        this.blocks[1](reconcile(blocks));
    }

    async saveValue(name: string, value: string): Promise<void> {
        const vals = await invoke<PlaceholderValue[]>("save_value", { name, value });
        this.values[1](reconcile(vals));
    }

    async deleteValue(name: string): Promise<void> {
        const vals = await invoke<PlaceholderValue[]>("delete_value", { name });
        this.values[1](reconcile(vals));
    }

    async renameValue(oldName: string, newName: string): Promise<void> {
        const vals = await invoke<PlaceholderValue[]>("rename_value", { oldName, newName });
        this.values[1](reconcile(vals));
    }

    async updateSettings(patch: Partial<Settings>): Promise<void> {
        const next: Settings = { ...this.settings[0], ...patch };
        const cfg = await invoke<Settings>("update_settings", { settings: next });
        this.settings[1](reconcile(cfg));
    }

    async reloadTasks(): Promise<void> {
        const tasks = await invoke<TaskMeta[]>("reload_tasks");
        this.tasks[1](reconcile(tasks));
    }

    async filterTaskSuggestions(input: string, cursor: number): Promise<void> {
        const sugs = await invoke<Suggestions>("filter_task_suggestions", { input, cursor });
        this.suggestions[1](reconcile(sugs));
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
            return null;
        } catch (e) {
            return String(e);
        }
    }

    async pasteValue(context: string, value: string): Promise<string> {
        const result = await invoke<PasteResult>("paste_value", { context, value });
        this.values[1](reconcile(result.values));
        return result.name;
    }
}
