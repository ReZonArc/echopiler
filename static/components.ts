// Copyright (c) 2021, Compiler Explorer Authors
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright notice,
//       this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

import {ParseFiltersAndOutputOptions} from '../types/features/filters.interfaces.js';
import {GccDumpViewState} from './panes/gccdump-view.interfaces.js';

import {ConfiguredOverrides} from './compilation/compiler-overrides.interfaces.js';
import {
    AST_VIEW_COMPONENT_NAME,
    CFG_VIEW_COMPONENT_NAME,
    CLANGIR_VIEW_COMPONENT_NAME,
    COMPILER_COMPONENT_NAME,
    CONFORMANCE_VIEW_COMPONENT_NAME,
    DEVICE_VIEW_COMPONENT_NAME,
    DIFF_VIEW_COMPONENT_NAME,
    EDITOR_COMPONENT_NAME,
    EXECUTOR_COMPONENT_NAME,
    FLAGS_VIEW_COMPONENT_NAME,
    GCC_DUMP_VIEW_COMPONENT_NAME,
    GNAT_DEBUG_TREE_VIEW_COMPONENT_NAME,
    GNAT_DEBUG_VIEW_COMPONENT_NAME,
    HASKELL_CMM_VIEW_COMPONENT_NAME,
    HASKELL_CORE_VIEW_COMPONENT_NAME,
    HASKELL_STG_VIEW_COMPONENT_NAME,
    IR_VIEW_COMPONENT_NAME,
    OPT_PIPELINE_VIEW_COMPONENT_NAME,
    OPT_VIEW_COMPONENT_NAME,
    OUTPUT_COMPONENT_NAME,
    PP_VIEW_COMPONENT_NAME,
    RUST_HIR_VIEW_COMPONENT_NAME,
    RUST_MACRO_EXP_VIEW_COMPONENT_NAME,
    RUST_MIR_VIEW_COMPONENT_NAME,
    STACK_USAGE_VIEW_COMPONENT_NAME,
    TOOL_COMPONENT_NAME,
    TOOL_INPUT_VIEW_COMPONENT_NAME,
    TREE_COMPONENT_NAME,
    TypedComponentConfig,
} from './components.interfaces.js';
import {ConfiguredRuntimeTools} from './execution/execution.interfaces.js';
import {LanguageKey} from './languages.interfaces.js';

/** Get an empty compiler component. */
export function getCompiler(editorId: number, lang: string): TypedComponentConfig<typeof COMPILER_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: COMPILER_COMPONENT_NAME,
        componentState: {
            source: editorId,
            lang,
        },
    };
}

/**
 * Get a compiler component with the given configuration.
 *
 * We have legacy calls in the codebase (to keep shortlinks permanent) that use this function without
 * passing langId and libs. To keep that support, the parameters are optional.
 *
 * TODO: Find the right type for options
 */
export function getCompilerWith(
    editorId: number,
    filters: ParseFiltersAndOutputOptions | undefined,
    options: unknown,
    compilerId: string,
    langId?: string,
    libs?: unknown,
): TypedComponentConfig<typeof COMPILER_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: COMPILER_COMPONENT_NAME,
        componentState: {
            source: editorId,
            compiler: compilerId,
            lang: langId,
            filters,
            options,
            libs,
        },
    };
}

/** Get a compiler for a tree mode component. */
export function getCompilerForTree(treeId: number, lang: string): TypedComponentConfig<typeof COMPILER_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: COMPILER_COMPONENT_NAME,
        componentState: {
            tree: treeId,
            lang,
        },
    };
}

/** Get an empty executor component. */
export function getExecutor(editorId: number, lang: string): TypedComponentConfig<typeof EXECUTOR_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: EXECUTOR_COMPONENT_NAME,
        componentState: {
            source: editorId,
            lang,
            compilationPanelShown: true,
            compilerOutShown: true,
        },
    };
}

/** Get a compiler component with the given configuration. */
export function getExecutorWith(
    editorId: number,
    lang: string,
    compilerId: string,
    libraries: {name: string; ver: string}[],
    compilerArgs: string | undefined,
    treeId: number,
    overrides?: ConfiguredOverrides,
    runtimeTools?: ConfiguredRuntimeTools,
): TypedComponentConfig<typeof EXECUTOR_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: EXECUTOR_COMPONENT_NAME,
        componentState: {
            source: editorId,
            tree: treeId,
            compiler: compilerId,
            libs: libraries,
            options: compilerArgs,
            lang,
            compilationPanelShown: true,
            compilerOutShown: true,
            overrides: overrides,
            runtimeTools: runtimeTools,
        },
    };
}

/** Get an executor for a tree mode component. */
export function getExecutorForTree(treeId: number, lang: string): TypedComponentConfig<typeof EXECUTOR_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: EXECUTOR_COMPONENT_NAME,
        componentState: {
            tree: treeId,
            lang,
            compilationPanelShown: true,
            compilerOutShown: true,
        },
    };
}

/**
 * Get an empty editor component.
 *
 * TODO: main.js calls this with no arguments.
 */
export function getEditor(langId: LanguageKey, id?: number): TypedComponentConfig<typeof EDITOR_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: EDITOR_COMPONENT_NAME,
        componentState: {
            id,
            lang: langId,
        },
    };
}

/** Get an editor component with the given configuration. */
export function getEditorWith(
    id: number,
    source: string,
    options: ParseFiltersAndOutputOptions,
    langId: string,
): TypedComponentConfig<typeof EDITOR_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: EDITOR_COMPONENT_NAME,
        componentState: {
            id,
            source,
            options,
            lang: langId,
        },
    };
}

/**
 * Get an empty tree component.
 *
 * TODO: main.js calls this with no arguments.
 */
export function getTree(id?: number): TypedComponentConfig<typeof TREE_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: TREE_COMPONENT_NAME,
        componentState: {
            id,
            cmakeArgs: '-DCMAKE_BUILD_TYPE=Debug',
        },
    };
}

/** Get an output component with the given configuration. */
export function getOutput(
    compiler: number,
    editor: number,
    tree: number,
): TypedComponentConfig<typeof OUTPUT_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: OUTPUT_COMPONENT_NAME,
        componentState: {
            compiler,
            editor,
            tree,
        },
    };
}

/** Get a tool view component with the given configuration. */
export function getToolViewWith(
    compilerId: number,
    compilerName: string,
    editorid: number,
    toolId: string,
    args: string,
    monacoStdin: boolean,
    tree: number,
): TypedComponentConfig<typeof TOOL_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: TOOL_COMPONENT_NAME,
        componentState: {
            id: compilerId,
            compilerName,
            editorid,
            toolId,
            args,
            tree,
            monacoStdin,
        },
    };
}

/** Get an empty tool input view component. */
export function getToolInputView(): TypedComponentConfig<typeof TOOL_INPUT_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: TOOL_INPUT_VIEW_COMPONENT_NAME,
        componentState: {},
    };
}

/** Get a tool input view component with the given configuration. */
export function getToolInputViewWith(
    compilerId: number,
    toolId: string,
    toolName: string,
): TypedComponentConfig<typeof TOOL_INPUT_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: TOOL_INPUT_VIEW_COMPONENT_NAME,
        componentState: {
            compilerId,
            toolId,
            toolName,
        },
    };
}

/** Get an empty diff component. */
export function getDiffView(): TypedComponentConfig<typeof DIFF_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: DIFF_VIEW_COMPONENT_NAME,
        componentState: {},
    };
}

/**
 * Get a diff component with the given configuration.
 *
 * TODO: possibly unused?
 */
export function getDiffViewWith(lhs: unknown, rhs: unknown): TypedComponentConfig<typeof DIFF_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: DIFF_VIEW_COMPONENT_NAME,
        componentState: {
            lhs,
            rhs,
        },
    };
}

/** Get an empty opt view component. */
export function getOptView(): TypedComponentConfig<typeof OPT_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: OPT_VIEW_COMPONENT_NAME,
        componentState: {},
    };
}

/** Get an opt view with the given configuration. */
export function getOptViewWith(
    id: number,
    source: string,
    optOutput: unknown,
    compilerName: string,
    editorid: number,
    treeid: number,
): TypedComponentConfig<typeof OPT_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: OPT_VIEW_COMPONENT_NAME,
        componentState: {
            id,
            source,
            optOutput,
            compilerName,
            editorid,
            treeid,
        },
    };
}

export function getStackUsageView(): TypedComponentConfig<typeof STACK_USAGE_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: STACK_USAGE_VIEW_COMPONENT_NAME,
        componentState: {},
    };
}
export function getStackUsageViewWith(
    id: number,
    source: string,
    suOutput: unknown,
    compilerName: string,
    editorid: number,
    treeid: number,
): TypedComponentConfig<typeof STACK_USAGE_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: STACK_USAGE_VIEW_COMPONENT_NAME,
        componentState: {
            id,
            source,
            suOutput,
            compilerName,
            editorid,
            treeid,
        },
    };
}

/** Get an empty flags view component. */
export function getFlagsView(): TypedComponentConfig<typeof FLAGS_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: FLAGS_VIEW_COMPONENT_NAME,
        componentState: {},
    };
}

/** Get a flags view with the given configuration. */
export function getFlagsViewWith(
    id: number,
    compilerName: string,
    compilerFlags: unknown,
): TypedComponentConfig<typeof FLAGS_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: FLAGS_VIEW_COMPONENT_NAME,
        componentState: {
            id,
            compilerName,
            compilerFlags,
        },
    };
}

/** Get an empty preprocessor view component. */
export function getPpView(): TypedComponentConfig<typeof PP_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: PP_VIEW_COMPONENT_NAME,
        componentState: {},
    };
}

/** Get a preprocessor view with the given configuration. */
export function getPpViewWith(
    id: number,
    source: string,
    ppOutput: unknown,
    compilerName: string,
    editorid: number,
    treeid: number,
): TypedComponentConfig<typeof PP_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: PP_VIEW_COMPONENT_NAME,
        componentState: {
            id,
            source,
            ppOutput,
            compilerName,
            editorid,
            treeid,
        },
    };
}

/** Get an empty ast view component. */
export function getAstView(): TypedComponentConfig<typeof AST_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: AST_VIEW_COMPONENT_NAME,
        componentState: {},
    };
}

/** Get an ast view with the given configuration. */
export function getAstViewWith(
    id: number,
    source: string,
    astOutput: unknown,
    compilerName: string,
    editorid: number,
    treeid: number,
): TypedComponentConfig<typeof AST_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: AST_VIEW_COMPONENT_NAME,
        componentState: {
            id,
            source,
            astOutput,
            compilerName,
            editorid,
            treeid,
        },
    };
}

/** Get an empty gcc dump view component. */
export function getGccDumpView(): TypedComponentConfig<typeof GCC_DUMP_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: GCC_DUMP_VIEW_COMPONENT_NAME,
        componentState: {},
    };
}

/** Get a gcc dump view with the given configuration. */
export function getGccDumpViewWith(
    id: number,
    compilerName: string,
    editorid: number,
    treeid: number,
    gccDumpOutput: GccDumpViewState,
): TypedComponentConfig<typeof GCC_DUMP_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: GCC_DUMP_VIEW_COMPONENT_NAME,
        componentState: {
            // PopulatedGccDumpViewState
            id,
            compilerName,
            editorid,
            treeid,

            // & GccDumpFiltersState
            ...gccDumpOutput,
        },
    };
}

/** Get an empty cfg view component. */
export function getCfgView(): TypedComponentConfig<typeof CFG_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: CFG_VIEW_COMPONENT_NAME,
        componentState: {},
    };
}

/** Get a cfg view with the given configuration. */
export function getCfgViewWith(
    id: number,
    editorid: number,
    treeid: number,
    isircfg?: boolean,
): TypedComponentConfig<typeof CFG_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: CFG_VIEW_COMPONENT_NAME,
        componentState: {
            selectedFunction: null,
            id,
            editorid,
            treeid,
            isircfg,
        },
    };
}

/** Get a conformance view with the given configuration. */
export function getConformanceView(
    editorid: number,
    treeid: number,
    source: string,
    langId: string,
): TypedComponentConfig<typeof CONFORMANCE_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: CONFORMANCE_VIEW_COMPONENT_NAME,
        componentState: {
            editorid,
            treeid,
            source,
            langId,
        },
    };
}

/** Get an empty ir view component. */
export function getIrView(): TypedComponentConfig<typeof IR_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: IR_VIEW_COMPONENT_NAME,
        componentState: {},
    };
}

/** Get a ir view with the given configuration. */
export function getIrViewWith(
    id: number,
    source: string,
    irOutput: unknown,
    compilerName: string,
    editorid: number,
    treeid: number,
): TypedComponentConfig<typeof IR_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: IR_VIEW_COMPONENT_NAME,
        componentState: {
            id,
            source,
            irOutput,
            compilerName,
            editorid,
            treeid,
        },
    };
}

export function getClangirView(): TypedComponentConfig<typeof CLANGIR_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: CLANGIR_VIEW_COMPONENT_NAME,
        componentState: {},
    };
}

export function getClangirViewWith(
    id: number,
    source: string,
    clangirOutput: unknown,
    compilerName: string,
    editorid: number,
    treeid: number,
): TypedComponentConfig<typeof CLANGIR_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: CLANGIR_VIEW_COMPONENT_NAME,
        componentState: {
            id,
            source,
            clangirOutput,
            compilerName,
            editorid,
            treeid,
        },
    };
}

/** Get an empty opt pipeline view component. */
export function getOptPipelineView(): TypedComponentConfig<typeof OPT_PIPELINE_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: OPT_PIPELINE_VIEW_COMPONENT_NAME,
        componentState: {},
    };
}

/** Get a opt pipeline view with the given configuration. */
export function getOptPipelineViewWith(
    id: number,
    lang: string,
    compilerId: string,
    compilerName: string,
    editorid: number,
    treeid: number,
): TypedComponentConfig<typeof OPT_PIPELINE_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: OPT_PIPELINE_VIEW_COMPONENT_NAME,
        componentState: {
            id,
            lang,
            compiler: compilerId,
            compilerName,
            editorid,
            treeid,
            selectedGroup: '',
            selectedIndex: 0,
            sidebarWidth: 0,
        },
    };
}

/** Get an empty rust mir view component. */
export function getRustMirView(): TypedComponentConfig<typeof RUST_MIR_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: RUST_MIR_VIEW_COMPONENT_NAME,
        componentState: {},
    };
}

/** Get a rust mir view with the given configuration. */
export function getRustMirViewWith(
    id: number,
    source: string,
    rustMirOutput: unknown,
    compilerName: string,
    editorid: number,
    treeid: number,
): TypedComponentConfig<typeof RUST_MIR_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: RUST_MIR_VIEW_COMPONENT_NAME,
        componentState: {
            id,
            source,
            rustMirOutput,
            compilerName,
            editorid,
            treeid,
        },
    };
}

/** Get an empty haskell core view component. */
export function getHaskellCoreView(): TypedComponentConfig<typeof HASKELL_CORE_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: HASKELL_CORE_VIEW_COMPONENT_NAME,
        componentState: {},
    };
}

/** Get a haskell core view with the given configuration. */
export function getHaskellCoreViewWith(
    id: number,
    source: string,
    haskellCoreOutput: unknown,
    compilerName: string,
    editorid: number,
    treeid: number,
): TypedComponentConfig<typeof HASKELL_CORE_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: HASKELL_CORE_VIEW_COMPONENT_NAME,
        componentState: {
            id,
            source,
            haskellCoreOutput,
            compilerName,
            editorid,
            treeid,
        },
    };
}

/** Get an empty haskell stg view component. */
export function getHaskellStgView(): TypedComponentConfig<typeof HASKELL_STG_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: HASKELL_STG_VIEW_COMPONENT_NAME,
        componentState: {},
    };
}

/** Get a haskell stg view with the given configuration. */
export function getHaskellStgViewWith(
    id: number,
    source: string,
    haskellStgOutput: unknown,
    compilerName: string,
    editorid: number,
    treeid: number,
): TypedComponentConfig<typeof HASKELL_STG_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: HASKELL_STG_VIEW_COMPONENT_NAME,
        componentState: {
            id,
            source,
            haskellStgOutput,
            compilerName,
            editorid,
            treeid,
        },
    };
}

/** Get an empty haskell cmm view component. */
export function getHaskellCmmView(): TypedComponentConfig<typeof HASKELL_CMM_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: HASKELL_CMM_VIEW_COMPONENT_NAME,
        componentState: {},
    };
}

export function getHaskellCmmViewWith(
    id: number,
    source: string,
    haskellCmmOutput: unknown,
    compilerName: string,
    editorid: number,
    treeid: number,
): TypedComponentConfig<typeof HASKELL_CMM_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: HASKELL_CMM_VIEW_COMPONENT_NAME,
        componentState: {
            id,
            source,
            haskellCmmOutput,
            compilerName,
            editorid,
            treeid,
        },
    };
}

/** Get an empty gnat debug tree view component. */
export function getGnatDebugTreeView(): TypedComponentConfig<typeof GNAT_DEBUG_TREE_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: GNAT_DEBUG_TREE_VIEW_COMPONENT_NAME,
        componentState: {},
    };
}

/** Get a gnat debug tree view with the given configuration. */
export function getGnatDebugTreeViewWith(
    id: number,
    source: string,
    gnatDebugTreeOutput: unknown,
    compilerName: string,
    editorid: number,
    treeid: number,
): TypedComponentConfig<typeof GNAT_DEBUG_TREE_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: GNAT_DEBUG_TREE_VIEW_COMPONENT_NAME,
        componentState: {
            id,
            source,
            gnatDebugTreeOutput,
            compilerName,
            editorid,
            treeid,
        },
    };
}

/** Get an empty gnat debug info view component. */
export function getGnatDebugView(): TypedComponentConfig<typeof GNAT_DEBUG_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: GNAT_DEBUG_VIEW_COMPONENT_NAME,
        componentState: {},
    };
}

/** Get a gnat debug info view with the given configuration. */
export function getGnatDebugViewWith(
    id: number,
    source: string,
    gnatDebugOutput: unknown,
    compilerName: string,
    editorid: number,
    treeid: number,
): TypedComponentConfig<typeof GNAT_DEBUG_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: GNAT_DEBUG_VIEW_COMPONENT_NAME,
        componentState: {
            id,
            source,
            gnatDebugOutput,
            compilerName,
            editorid,
            treeid,
        },
    };
}

/** Get an empty rust macro exp view component. */
export function getRustMacroExpView(): TypedComponentConfig<typeof RUST_MACRO_EXP_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: RUST_MACRO_EXP_VIEW_COMPONENT_NAME,
        componentState: {},
    };
}

/** Get a rust macro exp view with the given configuration. */
export function getRustMacroExpViewWith(
    id: number,
    source: string,
    rustMacroExpOutput: unknown,
    compilerName: string,
    editorid: number,
    treeid: number,
): TypedComponentConfig<typeof RUST_MACRO_EXP_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: RUST_MACRO_EXP_VIEW_COMPONENT_NAME,
        componentState: {
            id,
            source,
            rustMacroExpOutput,
            compilerName,
            editorid,
            treeid,
        },
    };
}

/** Get an empty rust hir view component. */
export function getRustHirView(): TypedComponentConfig<typeof RUST_HIR_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: RUST_HIR_VIEW_COMPONENT_NAME,
        componentState: {},
    };
}

/** Get a rust hir view with the given configuration. */
export function getRustHirViewWith(
    id: number,
    source: string,
    rustHirOutput: unknown,
    compilerName: string,
    editorid: number,
    treeid: number,
): TypedComponentConfig<typeof RUST_HIR_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: RUST_HIR_VIEW_COMPONENT_NAME,
        componentState: {
            id,
            source,
            rustHirOutput,
            compilerName,
            editorid,
            treeid,
        },
    };
}

/** Get an empty device view component. */
export function getDeviceView(): TypedComponentConfig<typeof DEVICE_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: DEVICE_VIEW_COMPONENT_NAME,
        componentState: {},
    };
}

/** Get a device view with the given configuration. */
export function getDeviceViewWith(
    id: number,
    source: string,
    devices: unknown,
    compilerName: string,
    editorid: number,
    treeid: number,
): TypedComponentConfig<typeof DEVICE_VIEW_COMPONENT_NAME> {
    return {
        type: 'component',
        componentName: DEVICE_VIEW_COMPONENT_NAME,
        componentState: {
            id: id,
            source: source,
            devices: devices,
            compilerName: compilerName,
            editorid: editorid,
            treeid: treeid,
        },
    };
}
