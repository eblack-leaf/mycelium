import {invoke} from "@tauri-apps/api/core";
import {createStore, SetStoreFunction, Store} from "solid-js/store";
import {Block} from "./bindings/Block.ts";
import {Suggestions} from "./bindings/Suggestions.ts";

export class Backend {
    blocks_store: [Store<Block[]>, SetStoreFunction<Block[]>];
    suggestions: [Store<Suggestions>, SetStoreFunction<Suggestions>];

    public constructor() {
        this.blocks_store = createStore<Block[]>([]);
        this.suggestions = createStore<Suggestions>({
            placeholders: [],
            ids: [],
            schema: [],
        });
    }

    public blocks() {
        return this.blocks_store[0]
    }

    async update() {
        this.blocks_store[1](await invoke("blocks"))
        this.suggestions[1](await invoke("suggestions"))
    }

    async add_block(text: string) {
        this.blocks_store[1]([...this.blocks_store[0], {text: text}]);
    }

    public placeholders() {
        return this.suggestions[0].placeholders
    }

    public ids() {
        return this.suggestions[0].ids
    }

    public schema() {
        return this.suggestions[0].schema
    }
}