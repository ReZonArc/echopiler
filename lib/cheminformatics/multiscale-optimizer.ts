/**
 * Multiscale Constraint Optimization Engine for Cosmeceutical Formulation
 * 
 * This module implements the core multiscale optimization engine that integrates
 * OpenCog-inspired reasoning with cosmeceutical formulation constraints. It addresses
 * simultaneous local and global optimization problems across different skin model
 * scales while maintaining regulatory compliance and maximizing therapeutic synergy.
 * 
 * Key Features:
 * - Hypergraph-based ingredient and action ontology encoding
 * - Symbolic and sub-symbolic optimization routines
 * - Multiscale constraint satisfaction across skin layers
 * - Recursive implementation pathways for complex formulations
 * - Integration with INCI search space reduction and attention allocation
 */

import {CosmeticIngredient, CosmeticFormulation} from '../../types/cheminformatics/cosmetic-chemistry.interfaces.js';
import {INCISearchSpaceReducer, SearchSpaceReductionConfig, MultiscaleOptimizationResult} from './inci-search-space-reducer.js';
import {AdaptiveAttentionAllocator, AttentionDistribution} from './adaptive-attention-allocator.js';

export interface SkinLayer {
    name: string;
    depth_range: {min: number; max: number}; // micrometers
    barrier_properties: {
        lipophilicity_requirement: number; // 0-1 scale
        molecular_weight_limit: number; // Daltons
        ph_tolerance: {min: number; max: number};
    };
    therapeutic_targets: string[];
    penetration_enhancers: string[];
    metabolic_activity: number; // 0-1 scale
}

export interface TherapeuticAction {
    id: string;
    name: string;
    mechanism: string;
    target_proteins: string[];
    target_pathways: string[];
    required_skin_layers: string[];
    concentration_response: {
        ec50: number; // effective concentration 50%
        hill_coefficient: number;
        max_effect: number;
    };
    time_course: {
        onset_hours: number;
        peak_hours: number;
        duration_hours: number;
    };
    synergy_potential: Map<string, number>;
}

export interface OptimizationConstraint {
    id: string;
    type: 'concentration' | 'compatibility' | 'regulatory' | 'stability' | 'cost' | 'synergy';
    priority: number; // 0-1, higher = more important
    hard_constraint: boolean; // true = must satisfy, false = prefer to satisfy
    scope: 'local' | 'global' | 'multiscale';
    description: string;
    evaluation_function: (formulation: CosmeticFormulation, context: OptimizationContext) => ConstraintEvaluation;
}

export interface ConstraintEvaluation {
    satisfied: boolean;
    satisfaction_degree: number; // 0-1, 1 = fully satisfied
    violation_magnitude: number; // 0+ higher = worse violation
    corrective_suggestions: string[];
}

export interface OptimizationContext {
    target_skin_type: string;
    environmental_conditions: Map<string, number>;
    user_preferences: Map<string, number>;
    regulatory_regions: string[];
    budget_constraints: {min: number; max: number};
    time_constraints: number; // days to market
    market_positioning: string;
}

export interface MultiscaleOptimizationConfig {
    max_iterations: number;
    convergence_threshold: number;
    exploration_probability: number;
    local_search_intensity: number;
    global_search_scope: number;
    constraint_penalty_weight: number;
    synergy_reward_weight: number;
    stability_weight: number;
    cost_weight: number;
    efficacy_weight: number;
}

export interface OptimizationResult {
    optimized_formulation: CosmeticFormulation;
    optimization_score: number;
    constraint_satisfaction: Map<string, ConstraintEvaluation>;
    therapeutic_efficacy: Map<string, number>;
    predicted_stability: number;
    estimated_cost: number;
    regulatory_compliance: Map<string, number>;
    synergy_matrix: Map<string, Map<string, number>>;
    optimization_trace: OptimizationStep[];
    convergence_metrics: {
        iterations_to_convergence: number;
        final_score: number;
        constraint_violations: number;
    };
}

export interface OptimizationStep {
    iteration: number;
    action: 'add_ingredient' | 'remove_ingredient' | 'adjust_concentration' | 'local_search' | 'global_jump';
    parameters: any;
    score_before: number;
    score_after: number;
    constraints_before: number;
    constraints_after: number;
    reasoning: string;
}

export class MultiscaleOptimizer {
    private skinModel: Map<string, SkinLayer>;
    private therapeuticActions: Map<string, TherapeuticAction>;
    private optimizationConstraints: Map<string, OptimizationConstraint>;
    private searchSpaceReducer: INCISearchSpaceReducer;
    private attentionAllocator: AdaptiveAttentionAllocator;

    constructor() {
        this.skinModel = new Map();
        this.therapeuticActions = new Map();
        this.optimizationConstraints = new Map();
        this.searchSpaceReducer = new INCISearchSpaceReducer();
        this.attentionAllocator = new AdaptiveAttentionAllocator();

        this.initializeSkinModel();
        this.initializeTherapeuticActions();
        this.initializeOptimizationConstraints();
    }

    /**
     * Initialize the multiscale skin model
     */
    private initializeSkinModel(): void {
        const layers: SkinLayer[] = [
            {
                name: 'stratum_corneum',
                depth_range: {min: 0, max: 20},
                barrier_properties: {
                    lipophilicity_requirement: 0.7,
                    molecular_weight_limit: 500,
                    ph_tolerance: {min: 4.5, max: 6.5}
                },
                therapeutic_targets: ['barrier_function', 'hydration', 'desquamation'],
                penetration_enhancers: ['urea', 'lactic_acid', 'ceramides'],
                metabolic_activity: 0.1
            },
            {
                name: 'viable_epidermis',
                depth_range: {min: 20, max: 100},
                barrier_properties: {
                    lipophilicity_requirement: 0.5,
                    molecular_weight_limit: 1000,
                    ph_tolerance: {min: 5.5, max: 7.5}
                },
                therapeutic_targets: ['cell_turnover', 'pigmentation', 'inflammation'],
                penetration_enhancers: ['glycolic_acid', 'retinoids', 'niacinamide'],
                metabolic_activity: 0.8
            },
            {
                name: 'papillary_dermis',
                depth_range: {min: 100, max: 500},
                barrier_properties: {
                    lipophilicity_requirement: 0.3,
                    molecular_weight_limit: 5000,
                    ph_tolerance: {min: 6.0, max: 8.0}
                },
                therapeutic_targets: ['collagen_synthesis', 'elastin_production', 'angiogenesis'],
                penetration_enhancers: ['peptides', 'growth_factors', 'microneedling'],
                metabolic_activity: 0.9
            },
            {
                name: 'reticular_dermis',
                depth_range: {min: 500, max: 3000},
                barrier_properties: {
                    lipophilicity_requirement: 0.2,
                    molecular_weight_limit: 10000,
                    ph_tolerance: {min: 6.5, max: 7.8}
                },
                therapeutic_targets: ['structural_support', 'wound_healing', 'tissue_remodeling'],
                penetration_enhancers: ['injectable_delivery', 'iontophoresis', 'ultrasound'],
                metabolic_activity: 0.6
            }
        ];

        layers.forEach(layer => {
            this.skinModel.set(layer.name, layer);
        });
    }

    /**
     * Initialize therapeutic actions database
     */
    private initializeTherapeuticActions(): void {
        const actions: TherapeuticAction[] = [
            {
                id: 'collagen_synthesis_stimulation',
                name: 'Collagen Synthesis Stimulation',
                mechanism: 'TGF-β pathway activation',
                target_proteins: ['COL1A1', 'COL3A1', 'TGFB1'],
                target_pathways: ['TGF-β signaling', 'mTOR pathway'],
                required_skin_layers: ['papillary_dermis', 'reticular_dermis'],
                concentration_response: {
                    ec50: 0.5, // % concentration
                    hill_coefficient: 2.0,
                    max_effect: 0.85
                },
                time_course: {
                    onset_hours: 72,
                    peak_hours: 168,
                    duration_hours: 336
                },
                synergy_potential: new Map([
                    ['vitamin_c', 0.8],
                    ['peptides', 0.9],
                    ['retinoids', 0.7]
                ])
            },
            {
                id: 'barrier_enhancement',
                name: 'Skin Barrier Enhancement',
                mechanism: 'Lipid bilayer reinforcement',
                target_proteins: ['FLG', 'LOR', 'IVL'],
                target_pathways: ['ceramide synthesis', 'tight junction formation'],
                required_skin_layers: ['stratum_corneum'],
                concentration_response: {
                    ec50: 2.0,
                    hill_coefficient: 1.5,
                    max_effect: 0.9
                },
                time_course: {
                    onset_hours: 4,
                    peak_hours: 24,
                    duration_hours: 72
                },
                synergy_potential: new Map([
                    ['ceramides', 0.9],
                    ['cholesterol', 0.8],
                    ['fatty_acids', 0.7]
                ])
            },
            {
                id: 'melanin_inhibition',
                name: 'Melanogenesis Inhibition',
                mechanism: 'Tyrosinase inhibition',
                target_proteins: ['TYR', 'TYRP1', 'DCT'],
                target_pathways: ['melanogenesis', 'MITF signaling'],
                required_skin_layers: ['viable_epidermis'],
                concentration_response: {
                    ec50: 1.0,
                    hill_coefficient: 1.8,
                    max_effect: 0.75
                },
                time_course: {
                    onset_hours: 48,
                    peak_hours: 120,
                    duration_hours: 240
                },
                synergy_potential: new Map([
                    ['niacinamide', 0.8],
                    ['arbutin', 0.7],
                    ['kojic_acid', 0.6]
                ])
            }
        ];

        actions.forEach(action => {
            this.therapeuticActions.set(action.id, action);
        });
    }

    /**
     * Initialize optimization constraints
     */
    private initializeOptimizationConstraints(): void {
        const constraints: OptimizationConstraint[] = [
            {
                id: 'total_actives_limit',
                type: 'concentration',
                priority: 0.9,
                hard_constraint: true,
                scope: 'global',
                description: 'Total active ingredient concentration must not exceed safe limits',
                evaluation_function: this.evaluateTotalActivesConstraint.bind(this)
            },
            {
                id: 'ingredient_compatibility',
                type: 'compatibility',
                priority: 0.95,
                hard_constraint: true,
                scope: 'global',
                description: 'All ingredients must be chemically compatible',
                evaluation_function: this.evaluateCompatibilityConstraint.bind(this)
            },
            {
                id: 'regulatory_compliance',
                type: 'regulatory',
                priority: 1.0,
                hard_constraint: true,
                scope: 'global',
                description: 'All ingredients must comply with regulatory requirements',
                evaluation_function: this.evaluateRegulatoryConstraint.bind(this)
            },
            {
                id: 'cost_effectiveness',
                type: 'cost',
                priority: 0.6,
                hard_constraint: false,
                scope: 'global',
                description: 'Formulation cost should be within budget constraints',
                evaluation_function: this.evaluateCostConstraint.bind(this)
            },
            {
                id: 'therapeutic_synergy',
                type: 'synergy',
                priority: 0.8,
                hard_constraint: false,
                scope: 'multiscale',
                description: 'Ingredients should exhibit synergistic therapeutic effects',
                evaluation_function: this.evaluateSynergyConstraint.bind(this)
            }
        ];

        constraints.forEach(constraint => {
            this.optimizationConstraints.set(constraint.id, constraint);
        });
    }

    /**
     * Main optimization function
     */
    public async optimizeFormulation(
        targetOutcomes: string[],
        context: OptimizationContext,
        config: MultiscaleOptimizationConfig
    ): Promise<OptimizationResult> {
        console.log('Starting multiscale cosmeceutical formulation optimization...');

        // Step 1: Reduce search space using INCI constraints
        const searchConfig: SearchSpaceReductionConfig = {
            max_ingredients: 12,
            max_total_actives_concentration: 25.0,
            target_therapeutic_vectors: targetOutcomes,
            skin_penetration_requirements: ['stratum_corneum', 'viable_epidermis'],
            stability_requirements: ['oxidation_resistant', 'ph_stable'],
            cost_constraints: context.budget_constraints,
            regulatory_regions: context.regulatory_regions
        };

        const searchSpaceResult = await this.searchSpaceReducer.reduceSearchSpace(
            {type: 'SKINCARE_FORMULATION'}, // Target formulation
            searchConfig
        );

        // Step 2: Update attention allocation
        const attentionDistribution = this.attentionAllocator.allocateAttention();
        
        // Step 3: Initialize formulation with most promising ingredients
        let currentFormulation = this.initializeFormulation(
            searchSpaceResult,
            attentionDistribution,
            context
        );

        // Step 4: Iterative optimization
        const optimizationTrace: OptimizationStep[] = [];
        let bestScore = -Infinity;
        let bestFormulation = currentFormulation;
        let iterations = 0;

        while (iterations < config.max_iterations) {
            const step = await this.performOptimizationStep(
                currentFormulation,
                searchSpaceResult,
                context,
                config,
                iterations
            );

            optimizationTrace.push(step);

            if (step.score_after > bestScore) {
                bestScore = step.score_after;
                bestFormulation = this.cloneFormulation(currentFormulation);
            }

            // Update attention based on step success
            this.attentionAllocator.reinforceAttention(
                `formulation_${iterations}`,
                step.score_after > step.score_before,
                step.score_after - step.score_before
            );

            // Check convergence
            if (this.checkConvergence(optimizationTrace, config.convergence_threshold)) {
                console.log(`Optimization converged after ${iterations + 1} iterations`);
                break;
            }

            iterations++;
        }

        // Step 5: Final evaluation and analysis
        const finalResult = await this.generateFinalResult(
            bestFormulation,
            context,
            optimizationTrace,
            iterations
        );

        console.log(`Optimization completed. Final score: ${finalResult.optimization_score}`);
        return finalResult;
    }

    /**
     * Initialize formulation with most promising ingredients
     */
    private initializeFormulation(
        searchSpaceResult: MultiscaleOptimizationResult,
        attentionDistribution: AttentionDistribution,
        context: OptimizationContext
    ): CosmeticFormulation {
        const formulation: CosmeticFormulation = {
            id: `optimized_${Date.now()}`,
            name: 'AI-Optimized Cosmeceutical Formulation',
            type: 'SKINCARE_FORMULATION',
            ingredients: [],
            concentrations: new Map(),
            total_cost: 0,
            ph_target: 6.0,
            stability_data: {
                formulation_id: `optimized_${Date.now()}`,
                stability_factors: [
                    {factor: 'ph_compatibility', risk_level: 'low'},
                    {factor: 'temperature_stability', risk_level: 'low'},
                    {factor: 'light_sensitivity', risk_level: 'medium'},
                    {factor: 'microbial_growth', risk_level: 'low'},
                    {factor: 'oxidation_risk', risk_level: 'medium'}
                ],
                shelf_life_estimate: 24,
                storage_conditions: [{
                    temperature_range: {min: 5, max: 25},
                    light_protection: true
                }],
                stability_rating: 'good'
            },
            regulatory_approvals: new Map(),
            target_properties: [],
            creation_date: new Date(),
            last_modified: new Date()
        };

        // Select top ingredients from reduced search space
        const topIngredients = searchSpaceResult.reduced_search_space.slice(0, 8);
        
        topIngredients.forEach(ingredient => {
            formulation.ingredients.push(ingredient);
            const concentration = searchSpaceResult.estimated_concentrations.get(ingredient.id) || 1.0;
            formulation.concentrations.set(ingredient.id, concentration);
            formulation.total_cost += (ingredient.cost_per_gram || 0.1) * concentration;
        });

        return formulation;
    }

    /**
     * Perform a single optimization step
     */
    private async performOptimizationStep(
        formulation: CosmeticFormulation,
        searchSpaceResult: MultiscaleOptimizationResult,
        context: OptimizationContext,
        config: MultiscaleOptimizationConfig,
        iteration: number
    ): Promise<OptimizationStep> {
        const scoreBefore = this.evaluateFormulation(formulation, context);
        const constraintsBefore = this.countConstraintViolations(formulation, context);

        let action: OptimizationStep['action'];
        let parameters: any = {};
        let reasoning = '';

        // Decide on optimization action based on attention and exploration
        const random = Math.random();
        
        if (random < config.exploration_probability) {
            // Global exploration
            if (Math.random() < 0.5) {
                action = 'add_ingredient';
                parameters = this.selectIngredientToAdd(formulation, searchSpaceResult, context);
                reasoning = 'Exploration: Adding new ingredient for potential synergy';
            } else {
                action = 'global_jump';
                parameters = this.performGlobalJump(formulation, searchSpaceResult, context);
                reasoning = 'Exploration: Global jump to escape local optimum';
            }
        } else {
            // Local optimization
            if (Math.random() < 0.6) {
                action = 'adjust_concentration';
                parameters = this.optimizeConcentrations(formulation, context);
                reasoning = 'Exploitation: Fine-tuning ingredient concentrations';
            } else if (Math.random() < 0.8) {
                action = 'local_search';
                parameters = this.performLocalSearch(formulation, context);
                reasoning = 'Exploitation: Local optimization of formulation';
            } else {
                action = 'remove_ingredient';
                parameters = this.selectIngredientToRemove(formulation, context);
                reasoning = 'Exploitation: Removing underperforming ingredient';
            }
        }

        // Apply the selected action
        this.applyOptimizationAction(formulation, action, parameters);

        const scoreAfter = this.evaluateFormulation(formulation, context);
        const constraintsAfter = this.countConstraintViolations(formulation, context);

        return {
            iteration,
            action,
            parameters,
            score_before: scoreBefore,
            score_after: scoreAfter,
            constraints_before: constraintsBefore,
            constraints_after: constraintsAfter,
            reasoning
        };
    }

    /**
     * Evaluate formulation quality score
     */
    private evaluateFormulation(
        formulation: CosmeticFormulation,
        context: OptimizationContext
    ): number {
        let totalScore = 0;
        let weightSum = 0;

        // Therapeutic efficacy score
        const efficacyScore = this.calculateTherapeuticEfficacy(formulation, context);
        totalScore += efficacyScore * 0.4;
        weightSum += 0.4;

        // Constraint satisfaction score
        const constraintScore = this.calculateConstraintSatisfaction(formulation, context);
        totalScore += constraintScore * 0.3;
        weightSum += 0.3;

        // Synergy score
        const synergyScore = this.calculateSynergyScore(formulation);
        totalScore += synergyScore * 0.2;
        weightSum += 0.2;

        // Cost-effectiveness score
        const costScore = this.calculateCostEffectiveness(formulation, context);
        totalScore += costScore * 0.1;
        weightSum += 0.1;

        return totalScore / weightSum;
    }

    /**
     * Calculate therapeutic efficacy across skin layers
     */
    private calculateTherapeuticEfficacy(
        formulation: CosmeticFormulation,
        context: OptimizationContext
    ): number {
        let totalEfficacy = 0;
        let actionCount = 0;

        this.therapeuticActions.forEach(action => {
            let actionEfficacy = 0;
            let contributingIngredients = 0;

            formulation.ingredients.forEach(ingredient => {
                if (ingredient.therapeutic_vectors?.some(vector => 
                    action.target_pathways.includes(vector) || 
                    action.target_proteins.includes(vector))) {
                    
                    const concentration = formulation.concentrations.get(ingredient.id) || 0;
                    const penetrationFactor = this.calculatePenetrationFactor(
                        ingredient, action.required_skin_layers
                    );
                    const doseResponse = this.calculateDoseResponse(
                        concentration, action.concentration_response
                    );
                    
                    actionEfficacy += doseResponse * penetrationFactor;
                    contributingIngredients++;
                }
            });

            if (contributingIngredients > 0) {
                totalEfficacy += actionEfficacy / contributingIngredients;
                actionCount++;
            }
        });

        return actionCount > 0 ? totalEfficacy / actionCount : 0;
    }

    /**
     * Calculate penetration factor for ingredient across skin layers
     */
    private calculatePenetrationFactor(
        ingredient: CosmeticIngredient,
        targetLayers: string[]
    ): number {
        let penetrationFactor = 1.0;

        targetLayers.forEach(layerName => {
            const layer = this.skinModel.get(layerName);
            if (layer) {
                // Check molecular weight constraint
                if (ingredient.molecularWeight && 
                    ingredient.molecularWeight > layer.barrier_properties.molecular_weight_limit) {
                    penetrationFactor *= 0.3; // Significant penalty
                }

                // Check pH compatibility
                if (ingredient.ph_stability_range) {
                    const phOverlap = this.calculateRangeOverlap(
                        ingredient.ph_stability_range,
                        layer.barrier_properties.ph_tolerance
                    );
                    penetrationFactor *= phOverlap;
                }

                // Check lipophilicity requirement
                const solubilityScore = this.calculateSolubilityScore(
                    ingredient.solubility,
                    layer.barrier_properties.lipophilicity_requirement
                );
                penetrationFactor *= solubilityScore;
            }
        });

        return Math.max(0.1, penetrationFactor); // Minimum 10% penetration
    }

    /**
     * Calculate dose-response relationship
     */
    private calculateDoseResponse(
        concentration: number,
        response: {ec50: number; hill_coefficient: number; max_effect: number}
    ): number {
        const hillEquation = (response.max_effect * Math.pow(concentration, response.hill_coefficient)) /
                            (Math.pow(response.ec50, response.hill_coefficient) + 
                             Math.pow(concentration, response.hill_coefficient));
        
        return Math.min(1.0, hillEquation);
    }

    /**
     * Calculate range overlap between two ranges
     */
    private calculateRangeOverlap(
        range1: {min: number; max: number},
        range2: {min: number; max: number}
    ): number {
        const overlapStart = Math.max(range1.min, range2.min);
        const overlapEnd = Math.min(range1.max, range2.max);
        
        if (overlapStart >= overlapEnd) return 0;
        
        const overlapSize = overlapEnd - overlapStart;
        const totalSize = Math.max(range1.max - range1.min, range2.max - range2.min);
        
        return overlapSize / totalSize;
    }

    /**
     * Calculate solubility compatibility score
     */
    private calculateSolubilityScore(
        ingredientSolubility: string,
        layerLipophilicity: number
    ): number {
        const solubilityScores = {
            'water_soluble': 1 - layerLipophilicity,
            'oil_soluble': layerLipophilicity,
            'both': 0.9,
            'insoluble': 0.2
        };

        return solubilityScores[ingredientSolubility as keyof typeof solubilityScores] || 0.5;
    }

    /**
     * Calculate constraint satisfaction score
     */
    private calculateConstraintSatisfaction(
        formulation: CosmeticFormulation,
        context: OptimizationContext
    ): number {
        let totalSatisfaction = 0;
        let totalWeight = 0;

        this.optimizationConstraints.forEach(constraint => {
            const evaluation = constraint.evaluation_function(formulation, context);
            const weight = constraint.priority * (constraint.hard_constraint ? 2 : 1);
            
            totalSatisfaction += evaluation.satisfaction_degree * weight;
            totalWeight += weight;
        });

        return totalWeight > 0 ? totalSatisfaction / totalWeight : 0;
    }

    /**
     * Calculate synergy score for formulation
     */
    private calculateSynergyScore(formulation: CosmeticFormulation): number {
        let totalSynergy = 0;
        let pairCount = 0;

        for (let i = 0; i < formulation.ingredients.length; i++) {
            for (let j = i + 1; j < formulation.ingredients.length; j++) {
                const ing1 = formulation.ingredients[i];
                const ing2 = formulation.ingredients[j];
                
                const synergy = this.calculatePairwiseSynergy(ing1, ing2);
                totalSynergy += synergy;
                pairCount++;
            }
        }

        return pairCount > 0 ? totalSynergy / pairCount : 0;
    }

    /**
     * Calculate pairwise synergy between ingredients
     */
    private calculatePairwiseSynergy(ing1: CosmeticIngredient, ing2: CosmeticIngredient): number {
        let synergy = 0.5; // Base neutral synergy

        // Check for known synergistic combinations
        this.therapeuticActions.forEach(action => {
            const synergy1 = action.synergy_potential.get(ing1.id) || 0;
            const synergy2 = action.synergy_potential.get(ing2.id) || 0;
            
            if (synergy1 > 0 && synergy2 > 0) {
                synergy += (synergy1 + synergy2) / 2 * 0.5;
            }
        });

        // Check for shared therapeutic vectors
        const sharedVectors = ing1.therapeutic_vectors?.filter(v =>
            ing2.therapeutic_vectors?.includes(v)
        ) || [];
        
        synergy += sharedVectors.length * 0.1;

        // Check for complementary mechanisms
        if (this.areComplementaryMechanisms(ing1, ing2)) {
            synergy += 0.2;
        }

        return Math.min(1.0, synergy);
    }

    /**
     * Check if ingredients have complementary mechanisms
     */
    private areComplementaryMechanisms(ing1: CosmeticIngredient, ing2: CosmeticIngredient): boolean {
        // Simple heuristic based on category and subtype combinations
        const complementaryPairs = [
            ['HUMECTANT', 'EMOLLIENT'],
            ['ACTIVE_INGREDIENT', 'PRESERVATIVE'],
            ['ANTIOXIDANT', 'UV_FILTER'],
            ['VITAMIN', 'PEPTIDE']
        ];

        return complementaryPairs.some(([type1, type2]) =>
            (ing1.subtype === type1 && ing2.subtype === type2) ||
            (ing1.subtype === type2 && ing2.subtype === type1)
        );
    }

    /**
     * Calculate cost-effectiveness score
     */
    private calculateCostEffectiveness(
        formulation: CosmeticFormulation,
        context: OptimizationContext
    ): number {
        const totalCost = formulation.total_cost;
        const maxBudget = context.budget_constraints.max;
        
        if (totalCost > maxBudget) {
            return 0; // Over budget
        }

        const costRatio = totalCost / maxBudget;
        return 1 - costRatio; // Lower cost = higher score
    }

    // Constraint evaluation functions
    private evaluateTotalActivesConstraint(
        formulation: CosmeticFormulation,
        context: OptimizationContext
    ): ConstraintEvaluation {
        const totalActives = Array.from(formulation.concentrations.values())
            .reduce((sum, conc) => sum + conc, 0);
        
        const maxAllowed = 25.0; // Maximum total actives percentage
        const satisfied = totalActives <= maxAllowed;
        const satisfactionDegree = satisfied ? 1.0 : maxAllowed / totalActives;
        const violationMagnitude = Math.max(0, totalActives - maxAllowed);

        return {
            satisfied,
            satisfaction_degree: satisfactionDegree,
            violation_magnitude: violationMagnitude,
            corrective_suggestions: satisfied ? [] : ['Reduce ingredient concentrations', 'Remove least effective ingredients']
        };
    }

    private evaluateCompatibilityConstraint(
        formulation: CosmeticFormulation,
        context: OptimizationContext
    ): ConstraintEvaluation {
        let incompatiblePairs = 0;
        const totalPairs = formulation.ingredients.length * (formulation.ingredients.length - 1) / 2;

        // Simple compatibility check (would be more sophisticated in practice)
        for (let i = 0; i < formulation.ingredients.length; i++) {
            for (let j = i + 1; j < formulation.ingredients.length; j++) {
                const ing1 = formulation.ingredients[i];
                const ing2 = formulation.ingredients[j];
                
                if (this.areIngredientsIncompatible(ing1, ing2)) {
                    incompatiblePairs++;
                }
            }
        }

        const satisfied = incompatiblePairs === 0;
        const satisfactionDegree = totalPairs > 0 ? 1 - (incompatiblePairs / totalPairs) : 1;

        return {
            satisfied,
            satisfaction_degree: satisfactionDegree,
            violation_magnitude: incompatiblePairs,
            corrective_suggestions: satisfied ? [] : ['Separate incompatible ingredients', 'Use stabilizing agents']
        };
    }

    private evaluateRegulatoryConstraint(
        formulation: CosmeticFormulation,
        context: OptimizationContext
    ): ConstraintEvaluation {
        let compliantIngredients = 0;
        
        formulation.ingredients.forEach(ingredient => {
            const isCompliant = context.regulatory_regions.every(region => {
                const status = ingredient.regulatory_status?.get(region);
                return status === 'approved';
            });
            
            if (isCompliant) compliantIngredients++;
        });

        const satisfied = compliantIngredients === formulation.ingredients.length;
        const satisfactionDegree = formulation.ingredients.length > 0 ? 
            compliantIngredients / formulation.ingredients.length : 1;

        return {
            satisfied,
            satisfaction_degree: satisfactionDegree,
            violation_magnitude: formulation.ingredients.length - compliantIngredients,
            corrective_suggestions: satisfied ? [] : ['Replace non-compliant ingredients', 'Seek regulatory approval']
        };
    }

    private evaluateCostConstraint(
        formulation: CosmeticFormulation,
        context: OptimizationContext
    ): ConstraintEvaluation {
        const totalCost = formulation.total_cost;
        const maxBudget = context.budget_constraints.max;
        
        const satisfied = totalCost <= maxBudget;
        const satisfactionDegree = satisfied ? 1.0 : maxBudget / totalCost;
        const violationMagnitude = Math.max(0, totalCost - maxBudget);

        return {
            satisfied,
            satisfaction_degree: satisfactionDegree,
            violation_magnitude: violationMagnitude,
            corrective_suggestions: satisfied ? [] : ['Reduce ingredient quantities', 'Find cost-effective alternatives']
        };
    }

    private evaluateSynergyConstraint(
        formulation: CosmeticFormulation,
        context: OptimizationContext
    ): ConstraintEvaluation {
        const synergyScore = this.calculateSynergyScore(formulation);
        const satisfied = synergyScore >= 0.6; // Threshold for good synergy
        
        return {
            satisfied,
            satisfaction_degree: synergyScore,
            violation_magnitude: Math.max(0, 0.6 - synergyScore),
            corrective_suggestions: satisfied ? [] : ['Add synergistic ingredient combinations', 'Remove antagonistic ingredients']
        };
    }

    // Utility methods for optimization actions
    private areIngredientsIncompatible(ing1: CosmeticIngredient, ing2: CosmeticIngredient): boolean {
        // Simplified incompatibility check
        const knownIncompatibilities = [
            ['retinol', 'vitamin_c'],
            ['benzoyl_peroxide', 'retinol'],
            ['aha', 'bha']
        ];

        return knownIncompatibilities.some(([id1, id2]) =>
            (ing1.id === id1 && ing2.id === id2) ||
            (ing1.id === id2 && ing2.id === id1)
        );
    }

    private countConstraintViolations(
        formulation: CosmeticFormulation,
        context: OptimizationContext
    ): number {
        let violations = 0;
        
        this.optimizationConstraints.forEach(constraint => {
            const evaluation = constraint.evaluation_function(formulation, context);
            if (!evaluation.satisfied) violations++;
        });

        return violations;
    }

    private selectIngredientToAdd(
        formulation: CosmeticFormulation,
        searchSpaceResult: MultiscaleOptimizationResult,
        context: OptimizationContext
    ): any {
        // Select ingredient not currently in formulation
        const availableIngredients = searchSpaceResult.reduced_search_space.filter(ing =>
            !formulation.ingredients.find(existing => existing.id === ing.id)
        );

        if (availableIngredients.length === 0) return null;

        // Score ingredients by potential synergy with existing formulation
        const scoredIngredients = availableIngredients.map(ingredient => {
            let synergyScore = 0;
            formulation.ingredients.forEach(existing => {
                synergyScore += this.calculatePairwiseSynergy(ingredient, existing);
            });
            
            return {ingredient, score: synergyScore / formulation.ingredients.length};
        });

        scoredIngredients.sort((a, b) => b.score - a.score);
        return {ingredient: scoredIngredients[0].ingredient, concentration: 1.0};
    }

    private selectIngredientToRemove(
        formulation: CosmeticFormulation,
        context: OptimizationContext
    ): any {
        if (formulation.ingredients.length <= 3) return null; // Keep minimum ingredients

        // Score ingredients by their contribution (lower score = remove first)
        const scoredIngredients = formulation.ingredients.map(ingredient => {
            const efficacyContribution = this.calculateIngredientEfficacyContribution(ingredient, formulation);
            const synergyContribution = this.calculateIngredientSynergyContribution(ingredient, formulation);
            const cost = (ingredient.cost_per_gram || 0.1) * (formulation.concentrations.get(ingredient.id) || 1);
            
            const score = (efficacyContribution + synergyContribution) / cost;
            return {ingredient, score};
        });

        scoredIngredients.sort((a, b) => a.score - b.score);
        return {ingredient: scoredIngredients[0].ingredient};
    }

    private calculateIngredientEfficacyContribution(
        ingredient: CosmeticIngredient,
        formulation: CosmeticFormulation
    ): number {
        // Calculate how much this ingredient contributes to overall efficacy
        let contribution = 0;
        
        this.therapeuticActions.forEach(action => {
            if (ingredient.therapeutic_vectors?.some(vector => 
                action.target_pathways.includes(vector))) {
                const concentration = formulation.concentrations.get(ingredient.id) || 0;
                const penetrationFactor = this.calculatePenetrationFactor(
                    ingredient, action.required_skin_layers
                );
                const doseResponse = this.calculateDoseResponse(
                    concentration, action.concentration_response
                );
                
                contribution += doseResponse * penetrationFactor;
            }
        });

        return contribution;
    }

    private calculateIngredientSynergyContribution(
        ingredient: CosmeticIngredient,
        formulation: CosmeticFormulation
    ): number {
        let synergyContribution = 0;
        
        formulation.ingredients.forEach(other => {
            if (other.id !== ingredient.id) {
                synergyContribution += this.calculatePairwiseSynergy(ingredient, other);
            }
        });

        return formulation.ingredients.length > 1 ? 
            synergyContribution / (formulation.ingredients.length - 1) : 0;
    }

    private optimizeConcentrations(
        formulation: CosmeticFormulation,
        context: OptimizationContext
    ): any {
        // Simple concentration adjustment
        const adjustments = new Map<string, number>();
        
        formulation.ingredients.forEach(ingredient => {
            const currentConc = formulation.concentrations.get(ingredient.id) || 1;
            const maxConc = ingredient.concentration_range?.max || 10;
            const minConc = ingredient.concentration_range?.min || 0.1;
            
            // Small random adjustment within constraints
            const adjustment = (Math.random() - 0.5) * 0.2 * currentConc;
            const newConc = Math.max(minConc, Math.min(maxConc, currentConc + adjustment));
            
            adjustments.set(ingredient.id, newConc);
            formulation.concentrations.set(ingredient.id, newConc);
        });

        return {adjustments};
    }

    private performLocalSearch(
        formulation: CosmeticFormulation,
        context: OptimizationContext
    ): any {
        // Implement local search heuristics
        return {action: 'local_optimization'};
    }

    private performGlobalJump(
        formulation: CosmeticFormulation,
        searchSpaceResult: MultiscaleOptimizationResult,
        context: OptimizationContext
    ): any {
        // Implement global jump strategy
        return {action: 'global_exploration'};
    }

    private applyOptimizationAction(
        formulation: CosmeticFormulation,
        action: OptimizationStep['action'], 
        parameters: any
    ): void {
        switch (action) {
            case 'add_ingredient':
                if (parameters?.ingredient) {
                    formulation.ingredients.push(parameters.ingredient);
                    formulation.concentrations.set(parameters.ingredient.id, parameters.concentration || 1.0);
                    formulation.total_cost += (parameters.ingredient.cost_per_gram || 0.1) * (parameters.concentration || 1.0);
                }
                break;
            case 'remove_ingredient':
                if (parameters?.ingredient) {
                    const index = formulation.ingredients.findIndex(ing => ing.id === parameters.ingredient.id);
                    if (index >= 0) {
                        formulation.ingredients.splice(index, 1);
                        const conc = formulation.concentrations.get(parameters.ingredient.id) || 0;
                        formulation.concentrations.delete(parameters.ingredient.id);
                        formulation.total_cost -= (parameters.ingredient.cost_per_gram || 0.1) * conc;
                    }
                }
                break;
            case 'adjust_concentration':
                if (parameters?.adjustments) {
                    parameters.adjustments.forEach((newConc: number, ingredientId: string) => {
                        const oldConc = formulation.concentrations.get(ingredientId) || 0;
                        formulation.concentrations.set(ingredientId, newConc);
                        
                        const ingredient = formulation.ingredients.find(ing => ing.id === ingredientId);
                        if (ingredient) {
                            formulation.total_cost += (ingredient.cost_per_gram || 0.1) * (newConc - oldConc);
                        }
                    });
                }
                break;
        }

        formulation.last_modified = new Date();
    }

    private checkConvergence(
        trace: OptimizationStep[],
        threshold: number
    ): boolean {
        if (trace.length < 10) return false;

        const recentSteps = trace.slice(-10);
        const scoreChanges = recentSteps.map(step => Math.abs(step.score_after - step.score_before));
        const avgChange = scoreChanges.reduce((a, b) => a + b, 0) / scoreChanges.length;

        return avgChange < threshold;
    }

    private cloneFormulation(formulation: CosmeticFormulation): CosmeticFormulation {
        return {
            ...formulation,
            ingredients: [...formulation.ingredients],
            concentrations: new Map(formulation.concentrations),
            regulatory_approvals: new Map(formulation.regulatory_approvals),
            target_properties: [...formulation.target_properties]
        };
    }

    private async generateFinalResult(
        formulation: CosmeticFormulation,
        context: OptimizationContext,
        trace: OptimizationStep[],
        iterations: number
    ): Promise<OptimizationResult> {
        const finalScore = this.evaluateFormulation(formulation, context);
        const constraintSatisfaction = new Map<string, ConstraintEvaluation>();
        
        this.optimizationConstraints.forEach((constraint, id) => {
            constraintSatisfaction.set(id, constraint.evaluation_function(formulation, context));
        });

        const therapeuticEfficacy = new Map<string, number>();
        this.therapeuticActions.forEach((action, id) => {
            therapeuticEfficacy.set(id, this.calculateTherapeuticEfficacy(formulation, context));
        });

        const synergyMatrix = new Map<string, Map<string, number>>();
        formulation.ingredients.forEach(ing1 => {
            const row = new Map<string, number>();
            formulation.ingredients.forEach(ing2 => {
                if (ing1.id !== ing2.id) {
                    row.set(ing2.id, this.calculatePairwiseSynergy(ing1, ing2));
                }
            });
            synergyMatrix.set(ing1.id, row);
        });

        const regulatoryCompliance = new Map<string, number>();
        context.regulatory_regions.forEach(region => {
            const compliance = formulation.ingredients.filter(ing =>
                ing.regulatory_status?.get(region) === 'approved'
            ).length / formulation.ingredients.length;
            regulatoryCompliance.set(region, compliance);
        });

        return {
            optimized_formulation: formulation,
            optimization_score: finalScore,
            constraint_satisfaction: constraintSatisfaction,
            therapeutic_efficacy: therapeuticEfficacy,
            predicted_stability: 0.85, // Placeholder
            estimated_cost: formulation.total_cost,
            regulatory_compliance: regulatoryCompliance,
            synergy_matrix: synergyMatrix,
            optimization_trace: trace,
            convergence_metrics: {
                iterations_to_convergence: iterations,
                final_score: finalScore,
                constraint_violations: this.countConstraintViolations(formulation, context)
            }
        };
    }
}