import "./App.css";
import * as Icon from "./components/feather.tsx";
import {createStore} from "solid-js/store";
import {BlockView, Block} from "./components/block_view.tsx";
import {For} from "solid-js";

type Session = {
    blocks: Block[],
    current: number
}
export default function App() {
    const session = createStore<Session>({
        blocks: [{
            query: "",
        }],
        current: 0
    });
    function set_query(str: string) {
        session[1]("blocks", [session[0].current], "query", str);
    }
    function get_query() {
        return session[0].blocks[session[0].current].query;
    }
    return (
        <main class="relative h-screen w-screen bg-stone-900 flex overflow-hidden gap-2 p-2">
            <div class={"flex-1 "}>
                <For each={session[0].blocks}>
                    {(bd) => <BlockView data={bd} getter={get_query} setter={set_query}/>}
                </For>
            </div>
            <div class={"w-10 flex flex-col gap-6"}>
                <div class={"h-8 flex-none"}>
                    <div class={"rounded-md bg-orange-400 flex items-center justify-center h-10"}>
                        <Icon.Terminal stroke={"#333333"} size={20}></Icon.Terminal>
                    </div>
                </div>
                <div class={"h-10 flex-none"}>
                    <div class={"rounded-md bg-stone-800 flex items-center justify-center h-10"}>
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
