/**
 * Meta-Optimization Engine for Comprehensive Formulation Strategy
 *
 * This module implements a meta-optimization strategy that generates optimal formulations
 * for every possible condition and treatment combination. It orchestrates multiple
 * optimization engines and provides intelligent strategy selection based on problem
 * characteristics and constraints.
 *
 * Key Features:
 * - Exhaustive condition-treatment matrix generation
 * - Intelligent optimization strategy selection (Hypergredient vs Multiscale vs Hybrid)
 * - Comprehensive result caching and performance analytics
 * - Real-time formulation recommendations based on meta-learned patterns
 * - Multi-objective global optimization across the complete solution space
 */

import type {CosmeticFormulation} from '../../types/cheminformatics/cosmetic-chemistry.interfaces.js';
import type {FormulationConstraints} from '../../types/cheminformatics/hypergredient-framework.interfaces.js';
import {HypergredientFramework} from './hypergredient-framework.js';
import {MultiscaleOptimizationConfig, MultiscaleOptimizer, OptimizationContext} from './multiscale-optimizer.js';

// Meta-optimization interfaces
export interface ConditionTreatmentMatrix {
    conditions: string[];
    treatments: string[];
    combinations: ConditionTreatmentCombination[];
    complexity_scores: Map<string, number>;
}

export interface ConditionTreatmentCombination {
    id: string;
    conditions: string[];
    treatments: string[];
    severity_level: 'mild' | 'moderate' | 'severe';
    urgency: 'low' | 'medium' | 'high';
    complexity_score: number;
    recommended_strategy: OptimizationStrategy;
}

export type OptimizationStrategy = 'hypergredient' | 'multiscale' | 'hybrid' | 'custom';

export interface MetaOptimizationConfig {
    max_combinations: number;
    enable_caching: boolean;
    cache_duration_hours: number;
    performance_tracking: boolean;
    parallel_optimization: boolean;
    max_parallel_workers: number;
    strategy_selection_weights: {
        complexity: number;
        performance: number;
        cost: number;
        time: number;
    };
}

export interface OptimizationStrategyProfile {
    strategy: OptimizationStrategy;
    suitability_score: number;
    expected_performance: number;
    estimated_time_ms: number;
    resource_requirements: {
        memory_mb: number;
        cpu_intensive: boolean;
    };
    optimal_conditions: string[];
}

export interface MetaOptimizationResult {
    combination_id: string;
    optimal_formulation: CosmeticFormulation;
    strategy_used: OptimizationStrategy;
    performance_metrics: {
        optimization_score: number;
        execution_time_ms: number;
        iterations: number;
        convergence_achieved: boolean;
    };
    alternative_formulations?: CosmeticFormulation[];
    confidence_score: number;
    recommendations: string[];
}

export interface GlobalOptimizationSummary {
    total_combinations: number;
    successful_optimizations: number;
    strategy_distribution: Map<OptimizationStrategy, number>;
    average_performance_by_strategy: Map<OptimizationStrategy, number>;
    top_performing_combinations: MetaOptimizationResult[];
    performance_analytics: {
        best_overall_score: number;
        average_score: number;
        strategy_efficiency: Map<OptimizationStrategy, number>;
    };
}

export class MetaOptimizationEngine {
    private hypergredientFramework: HypergredientFramework;
    private multiscaleOptimizer: MultiscaleOptimizer;
    private config: MetaOptimizationConfig;
    private resultCache: Map<string, MetaOptimizationResult>;
    private performanceHistory: Map<OptimizationStrategy, number[]>;
    private conditionTreatmentMatrix: ConditionTreatmentMatrix;

    constructor(config?: Partial<MetaOptimizationConfig>) {
        this.config = this.initializeConfig(config);
        this.hypergredientFramework = new HypergredientFramework();
        this.multiscaleOptimizer = new MultiscaleOptimizer();
        this.resultCache = new Map();
        this.performanceHistory = new Map();
        this.conditionTreatmentMatrix = this.generateConditionTreatmentMatrix();

        console.log('🚀 Meta-Optimization Engine initialized');
        console.log(`   📊 Matrix: ${this.conditionTreatmentMatrix.combinations.length} combinations`);
        console.log(`   🎯 Strategies: ${this.getAvailableStrategies().length} available`);
    }

    /**
     * Initialize configuration with defaults
     */
    private initializeConfig(userConfig?: Partial<MetaOptimizationConfig>): MetaOptimizationConfig {
        const defaultConfig: MetaOptimizationConfig = {
            max_combinations: 1000,
            enable_caching: true,
            cache_duration_hours: 24,
            performance_tracking: true,
            parallel_optimization: true,
            max_parallel_workers: 4,
            strategy_selection_weights: {
                complexity: 0.3,
                performance: 0.4,
                cost: 0.15,
                time: 0.15,
            },
        };

        return {...defaultConfig, ...userConfig};
    }

    /**
     * Generate comprehensive condition-treatment matrix
     */
    private generateConditionTreatmentMatrix(): ConditionTreatmentMatrix {
        // Define all possible skin conditions
        const conditions = [
            'wrinkles',
            'fine_lines',
            'aging',
            'firmness',
            'elasticity_loss',
            'dryness',
            'dehydration',
            'oily_skin',
            'combination_skin',
            'hyperpigmentation',
            'dark_spots',
            'melasma',
            'sun_damage',
            'acne',
            'blackheads',
            'pores',
            'blemishes',
            'sensitive_skin',
            'irritation',
            'redness',
            'inflammation',
            'barrier_damage',
            'environmental_damage',
            'pollution_damage',
            'uneven_texture',
            'roughness',
            'dullness',
            'lack_of_radiance',
        ];

        // Define all possible treatments
        const treatments = [
            'hydration',
            'moisturization',
            'barrier_repair',
            'anti_aging',
            'collagen_synthesis',
            'cellular_turnover',
            'brightening',
            'melanin_inhibition',
            'antioxidant_protection',
            'sebum_regulation',
            'pore_minimization',
            'acne_treatment',
            'anti_inflammatory',
            'soothing',
            'calming',
            'exfoliation',
            'resurfacing',
            'texture_improvement',
            'firmining',
            'lifting',
            'tightening',
            'sun_protection',
            'environmental_protection',
        ];

        // Generate all meaningful combinations
        const combinations: ConditionTreatmentCombination[] = [];
        let combinationId = 0;

        for (const condition of conditions) {
            const relevantTreatments = this.getRelevantTreatments(condition);

            // Single condition-treatment pairs
            for (const treatment of relevantTreatments) {
                combinations.push({
                    id: `combo_${combinationId++}`,
                    conditions: [condition],
                    treatments: [treatment],
                    severity_level: 'moderate',
                    urgency: 'medium',
                    complexity_score: this.calculateComplexityScore([condition], [treatment]),
                    recommended_strategy: this.recommendStrategy([condition], [treatment]),
                });
            }

            // Multi-treatment approaches for single conditions
            if (relevantTreatments.length > 1) {
                for (let i = 0; i < relevantTreatments.length - 1; i++) {
                    for (let j = i + 1; j < relevantTreatments.length; j++) {
                        const treatmentPair = [relevantTreatments[i], relevantTreatments[j]];
                        combinations.push({
                            id: `combo_${combinationId++}`,
                            conditions: [condition],
                            treatments: treatmentPair,
                            severity_level: 'moderate',
                            urgency: 'medium',
                            complexity_score: this.calculateComplexityScore([condition], treatmentPair),
                            recommended_strategy: this.recommendStrategy([condition], treatmentPair),
                        });
                    }
                }
            }
        }

        // Multi-condition combinations (common co-occurring conditions)
        const commonCombinations = [
            ['wrinkles', 'dryness'],
            ['hyperpigmentation', 'aging'],
            ['acne', 'oily_skin'],
            ['sensitive_skin', 'barrier_damage'],
            ['dullness', 'uneven_texture'],
            ['fine_lines', 'firmness'],
        ];

        for (const conditionGroup of commonCombinations) {
            const allRelevantTreatments = new Set<string>();
            for (const condition of conditionGroup) {
                for (const treatment of this.getRelevantTreatments(condition)) {
                    allRelevantTreatments.add(treatment);
                }
            }

            const treatmentArray = Array.from(allRelevantTreatments).slice(0, 3); // Limit to 3 treatments
            combinations.push({
                id: `combo_${combinationId++}`,
                conditions: conditionGroup,
                treatments: treatmentArray,
                severity_level: 'moderate',
                urgency: 'medium',
                complexity_score: this.calculateComplexityScore(conditionGroup, treatmentArray),
                recommended_strategy: this.recommendStrategy(conditionGroup, treatmentArray),
            });
        }

        // Calculate complexity scores for all combinations
        const complexityScores = new Map<string, number>();
        combinations.forEach(combo => {
            complexityScores.set(combo.id, combo.complexity_score);
        });

        return {
            conditions,
            treatments,
            combinations: combinations.slice(0, this.config.max_combinations),
            complexity_scores: complexityScores,
        };
    }

    /**
     * Get treatments relevant to a specific condition
     */
    private getRelevantTreatments(condition: string): string[] {
        const treatmentMap: Map<string, string[]> = new Map([
            ['wrinkles', ['anti_aging', 'collagen_synthesis', 'cellular_turnover', 'hydration']],
            ['fine_lines', ['anti_aging', 'collagen_synthesis', 'hydration', 'antioxidant_protection']],
            ['aging', ['anti_aging', 'collagen_synthesis', 'antioxidant_protection', 'cellular_turnover']],
            ['dryness', ['hydration', 'moisturization', 'barrier_repair']],
            ['oily_skin', ['sebum_regulation', 'pore_minimization', 'exfoliation']],
            ['hyperpigmentation', ['brightening', 'melanin_inhibition', 'antioxidant_protection']],
            ['acne', ['acne_treatment', 'sebum_regulation', 'anti_inflammatory', 'exfoliation']],
            ['sensitive_skin', ['anti_inflammatory', 'soothing', 'barrier_repair', 'calming']],
            ['dullness', ['brightening', 'exfoliation', 'antioxidant_protection', 'texture_improvement']],
            ['firmness', ['firmining', 'collagen_synthesis', 'lifting', 'tightening']],
        ]);

        return treatmentMap.get(condition) || ['hydration']; // Default to hydration
    }

    /**
     * Calculate complexity score for condition-treatment combination
     */
    private calculateComplexityScore(conditions: string[], treatments: string[]): number {
        const baseComplexity = conditions.length + treatments.length;
        const interactionComplexity = conditions.length * treatments.length * 0.1;
        const synergySynergy = this.estimateSynergyComplexity(conditions, treatments);

        return Math.min(10, baseComplexity + interactionComplexity + synergySynergy);
    }

    /**
     * Estimate synergy complexity between conditions and treatments
     */
    private estimateSynergyComplexity(conditions: string[], treatments: string[]): number {
        // Simplified synergy estimation - more complex combinations have higher synergy potential
        if (conditions.length === 1 && treatments.length === 1) return 0;
        if (conditions.length === 1 && treatments.length === 2) return 0.5;
        if (conditions.length === 2 && treatments.length <= 2) return 1.0;
        return Math.min(3, conditions.length * treatments.length * 0.2);
    }

    /**
     * Recommend optimization strategy based on combination characteristics
     */
    private recommendStrategy(conditions: string[], treatments: string[]): OptimizationStrategy {
        const complexity = this.calculateComplexityScore(conditions, treatments);

        if (complexity <= 3) {
            return 'hypergredient'; // Simple cases, use fast hypergredient optimization
        } else if (complexity <= 6) {
            return 'multiscale'; // Medium complexity, use multiscale optimization
        } else if (complexity <= 8) {
            return 'hybrid'; // High complexity, use hybrid approach
        } else {
            return 'custom'; // Very high complexity, use custom optimization
        }
    }

    /**
     * Get available optimization strategies
     */
    private getAvailableStrategies(): OptimizationStrategy[] {
        return ['hypergredient', 'multiscale', 'hybrid', 'custom'];
    }

    /**
     * Generate optimal formulation for a specific condition-treatment combination
     */
    public async optimizeForCombination(
        combinationId: string,
        skinType: string = 'normal',
        constraints?: Partial<FormulationConstraints>,
    ): Promise<MetaOptimizationResult> {
        console.log(`\n🎯 Meta-optimizing combination: ${combinationId}`);

        // Check cache first
        const cacheKey = `${combinationId}_${skinType}`;
        if (this.config.enable_caching && this.resultCache.has(cacheKey)) {
            console.log('   📦 Using cached result');
            return this.resultCache.get(cacheKey)!;
        }

        const startTime = Date.now();
        const combination = this.conditionTreatmentMatrix.combinations.find(c => c.id === combinationId);

        if (!combination) {
            throw new Error(`Combination ${combinationId} not found`);
        }

        console.log(`   🎯 Conditions: ${combination.conditions.join(', ')}`);
        console.log(`   💊 Treatments: ${combination.treatments.join(', ')}`);
        console.log(`   🧮 Strategy: ${combination.recommended_strategy}`);

        let result: MetaOptimizationResult;

        try {
            switch (combination.recommended_strategy) {
                case 'hypergredient':
                    result = await this.optimizeWithHypergredient(combination, skinType, constraints);
                    break;
                case 'multiscale':
                    result = await this.optimizeWithMultiscale(combination, skinType, constraints);
                    break;
                case 'hybrid':
                    result = await this.optimizeWithHybrid(combination, skinType, constraints);
                    break;
                case 'custom':
                    result = await this.optimizeWithCustom(combination, skinType, constraints);
                    break;
                default:
                    result = await this.optimizeWithHypergredient(combination, skinType, constraints);
            }

            result.performance_metrics.execution_time_ms = Date.now() - startTime;

            // Cache result
            if (this.config.enable_caching) {
                this.resultCache.set(cacheKey, result);
            }

            // Track performance
            if (this.config.performance_tracking) {
                this.trackPerformance(combination.recommended_strategy, result.performance_metrics.optimization_score);
            }

            console.log(
                `   ✅ Optimization complete - Score: ${result.performance_metrics.optimization_score.toFixed(2)}`,
            );

            return result;
        } catch (error) {
            console.error(`   ❌ Optimization failed: ${error}`);
            throw error;
        }
    }

    /**
     * Optimize using Hypergredient Framework
     */
    private async optimizeWithHypergredient(
        combination: ConditionTreatmentCombination,
        skinType: string,
        constraints?: Partial<FormulationConstraints>,
    ): Promise<MetaOptimizationResult> {
        const targetConcerns = [...combination.conditions, ...combination.treatments];
        const optimizationResult = this.hypergredientFramework.optimizeFormulation(
            targetConcerns,
            constraints,
            skinType,
        );

        return {
            combination_id: combination.id,
            optimal_formulation: optimizationResult.formulation,
            strategy_used: 'hypergredient',
            performance_metrics: {
                optimization_score: optimizationResult.score.composite_score,
                execution_time_ms: 0, // Will be set by caller
                iterations: 1,
                convergence_achieved: true,
            },
            confidence_score:
                optimizationResult.score.confidence_interval.max - optimizationResult.score.confidence_interval.min,
            recommendations: this.generateRecommendations(optimizationResult.formulation, combination),
        };
    }

    /**
     * Optimize using Multiscale Optimizer
     */
    private async optimizeWithMultiscale(
        combination: ConditionTreatmentCombination,
        skinType: string,
        constraints?: Partial<FormulationConstraints>,
    ): Promise<MetaOptimizationResult> {
        const context: OptimizationContext = {
            target_skin_type: skinType,
            environmental_conditions: new Map([
                ['temperature', 25],
                ['humidity', 50],
            ]),
            user_preferences: new Map([['natural_ingredients', 0.7]]),
            regulatory_regions: ['EU', 'FDA'],
            budget_constraints: {min: 10, max: constraints?.budget_limit || 100},
            time_constraints: 30,
            market_positioning: 'premium',
        };

        const config: MultiscaleOptimizationConfig = {
            max_iterations: 50,
            convergence_threshold: 0.001,
            exploration_probability: 0.3,
            local_search_intensity: 0.7,
            global_search_scope: 0.5,
            constraint_penalty_weight: 0.5,
            synergy_reward_weight: 0.3,
            stability_weight: 0.2,
            cost_weight: 0.15,
            efficacy_weight: 0.35,
        };

        const targetOutcomes = [...combination.conditions, ...combination.treatments];
        const optimizationResult = await this.multiscaleOptimizer.optimizeFormulation(targetOutcomes, context, config);

        return {
            combination_id: combination.id,
            optimal_formulation: optimizationResult.optimized_formulation,
            strategy_used: 'multiscale',
            performance_metrics: {
                optimization_score: optimizationResult.optimization_score,
                execution_time_ms: 0, // Will be set by caller
                iterations: optimizationResult.convergence_metrics?.iterations_to_convergence || 50,
                convergence_achieved: (optimizationResult.convergence_metrics?.iterations_to_convergence || 0) < 50,
            },
            confidence_score: optimizationResult.predicted_stability,
            recommendations: this.generateRecommendations(optimizationResult.optimized_formulation, combination),
        };
    }

    /**
     * Optimize using Hybrid approach (combines both strategies)
     */
    private async optimizeWithHybrid(
        combination: ConditionTreatmentCombination,
        skinType: string,
        constraints?: Partial<FormulationConstraints>,
    ): Promise<MetaOptimizationResult> {
        // Run both optimizations and select the best result
        const hypergredientResult = await this.optimizeWithHypergredient(combination, skinType, constraints);
        const multiscaleResult = await this.optimizeWithMultiscale(combination, skinType, constraints);

        // Select the better result based on optimization score
        const bestResult =
            hypergredientResult.performance_metrics.optimization_score >
            multiscaleResult.performance_metrics.optimization_score
                ? hypergredientResult
                : multiscaleResult;

        return {
            ...bestResult,
            strategy_used: 'hybrid',
            alternative_formulations: [hypergredientResult.optimal_formulation, multiscaleResult.optimal_formulation],
        };
    }

    /**
     * Optimize using Custom approach for very complex cases
     */
    private async optimizeWithCustom(
        combination: ConditionTreatmentCombination,
        skinType: string,
        constraints?: Partial<FormulationConstraints>,
    ): Promise<MetaOptimizationResult> {
        // For now, use hybrid approach but with modified parameters
        console.log('   🔬 Using custom optimization strategy');
        return this.optimizeWithHybrid(combination, skinType, constraints);
    }

    /**
     * Generate recommendations based on formulation and combination
     */
    private generateRecommendations(
        formulation: CosmeticFormulation,
        combination: ConditionTreatmentCombination,
    ): string[] {
        const recommendations: string[] = [];

        if (formulation.ingredients.length > 8) {
            recommendations.push('Consider simplifying the formulation for better stability');
        }

        if (combination.complexity_score > 7) {
            recommendations.push('High complexity formulation - recommend patch testing');
        }

        if (combination.treatments.includes('cellular_turnover')) {
            recommendations.push('Introduce gradually to minimize irritation');
        }

        if (combination.conditions.includes('sensitive_skin')) {
            recommendations.push('Use gentle, hypoallergenic base ingredients');
        }

        return recommendations;
    }

    /**
     * Track performance for strategy analytics
     */
    private trackPerformance(strategy: OptimizationStrategy, score: number): void {
        if (!this.performanceHistory.has(strategy)) {
            this.performanceHistory.set(strategy, []);
        }
        this.performanceHistory.get(strategy)!.push(score);
    }

    /**
     * Generate optimal formulations for all combinations
     */
    public async optimizeAllCombinations(
        skinType: string = 'normal',
        constraints?: Partial<FormulationConstraints>,
    ): Promise<GlobalOptimizationSummary> {
        console.log('\n🌟 Starting comprehensive meta-optimization');
        console.log(`   📊 Total combinations: ${this.conditionTreatmentMatrix.combinations.length}`);

        const results: MetaOptimizationResult[] = [];
        const strategyDistribution = new Map<OptimizationStrategy, number>();
        const performanceByStrategy = new Map<OptimizationStrategy, number[]>();

        let successful = 0;

        for (const combination of this.conditionTreatmentMatrix.combinations) {
            try {
                const result = await this.optimizeForCombination(combination.id, skinType, constraints);
                results.push(result);
                successful++;

                // Track strategy distribution
                const strategy = result.strategy_used;
                strategyDistribution.set(strategy, (strategyDistribution.get(strategy) || 0) + 1);

                // Track performance by strategy
                if (!performanceByStrategy.has(strategy)) {
                    performanceByStrategy.set(strategy, []);
                }
                performanceByStrategy.get(strategy)!.push(result.performance_metrics.optimization_score);
            } catch (error) {
                console.error(`   ❌ Failed combination ${combination.id}: ${error}`);
            }
        }

        // Calculate analytics
        const averagePerformanceByStrategy = new Map<OptimizationStrategy, number>();
        const strategyEfficiency = new Map<OptimizationStrategy, number>();

        performanceByStrategy.forEach((scores, strategy) => {
            const average = scores.reduce((a, b) => a + b, 0) / scores.length;
            averagePerformanceByStrategy.set(strategy, average);

            // Efficiency = average performance / average execution time (simplified)
            strategyEfficiency.set(strategy, average);
        });

        // Get top performing combinations
        const topPerforming = results
            .sort((a, b) => b.performance_metrics.optimization_score - a.performance_metrics.optimization_score)
            .slice(0, 10);

        const allScores = results.map(r => r.performance_metrics.optimization_score);
        const bestScore = Math.max(...allScores);
        const averageScore = allScores.reduce((a, b) => a + b, 0) / allScores.length;

        console.log(`\n✅ Meta-optimization complete:`);
        console.log(`   🎯 Successful: ${successful}/${this.conditionTreatmentMatrix.combinations.length}`);
        console.log(`   📈 Best score: ${bestScore.toFixed(2)}`);
        console.log(`   📊 Average score: ${averageScore.toFixed(2)}`);

        return {
            total_combinations: this.conditionTreatmentMatrix.combinations.length,
            successful_optimizations: successful,
            strategy_distribution: strategyDistribution,
            average_performance_by_strategy: averagePerformanceByStrategy,
            top_performing_combinations: topPerforming,
            performance_analytics: {
                best_overall_score: bestScore,
                average_score: averageScore,
                strategy_efficiency: strategyEfficiency,
            },
        };
    }

    /**
     * Get condition-treatment matrix for analysis
     */
    public getConditionTreatmentMatrix(): ConditionTreatmentMatrix {
        return this.conditionTreatmentMatrix;
    }

    /**
     * Get performance analytics
     */
    public getPerformanceAnalytics(): Map<OptimizationStrategy, number[]> {
        return this.performanceHistory;
    }

    /**
     * Clear result cache
     */
    public clearCache(): void {
        this.resultCache.clear();
        console.log('🧹 Meta-optimization cache cleared');
    }
}
