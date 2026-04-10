import "./App.css";
import * as Icon from "./components/feather.tsx";
import {BlockView} from "./components/block_view.tsx";
import {For} from "solid-js";
import {Backend} from "./backend.tsx";

export default function App() {
    const backend = new Backend();
    return (
        <main class="relative h-screen w-screen bg-stone-900 flex overflow-hidden gap-2 p-2">
            <div class={"flex-1"}>
                <For each={backend.blocks()}>
                    {(block) => <BlockView block={block} backend={backend}/>}
                </For>
            </div>
            <div class={"w-10 flex flex-col gap-4"}>
                <div class={"h-8 flex-none"}>
                    <div class={"rounded-md bg-orange-400 flex items-center justify-center h-10"}
                         onClick={async () => {
                             await backend.update();
                         }}>
                        <Icon.Terminal stroke={"#333333"} size={20}></Icon.Terminal>
                    </div>
                </div>
                <div class={"h-10 flex-none"}>
                    <div class={"rounded-md bg-stone-800 flex items-center justify-center h-10"}
                         onClick={async () => {
                             await backend.add_block("hello world");
                         }}>
                        <Icon.ChevronDown size={20} stroke={"#d4d4d4"}></Icon.ChevronDown>
                    </div>
                </div>
                <div class={"flex-1 flex items-end  justify-center"}>
                    <div class={"rounded-full bg-stone-800 flex items-center justify-center h-10 w-10"}>
                        <Icon.Settings stroke={"#888888"} size={20}></Icon.Settings>
                    </div>
                </div>
            </div>
        </main>
    );
}
