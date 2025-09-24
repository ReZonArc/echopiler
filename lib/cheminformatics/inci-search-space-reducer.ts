/**
 * INCI-Driven Search Space Reduction for Cosmeceutical Formulation
 * 
 * This module implements advanced algorithms for reducing the cosmetic ingredient
 * search space based on INCI (International Nomenclature of Cosmetic Ingredients)
 * constraints, regulatory compliance, and multiscale optimization requirements.
 * 
 * Key Features:
 * - INCI subset constraint satisfaction
 * - Concentration estimation from regulatory ordering
 * - Multiscale therapeutic vector optimization
 * - Regulatory compliance validation
 * - Synergistic ingredient discovery
 */

import {CosmeticIngredient, CosmeticFormulation, IngredientCategory} from '../../types/cheminformatics/cosmetic-chemistry.interfaces.js';

export interface INCIConstraint {
    approved_ingredients: Set<string>;
    concentration_limits: Map<string, {min: number; max: number}>;
    incompatible_pairs: Set<string>;
    regulatory_region: string;
    allergen_threshold: number;
}

export interface SearchSpaceReductionConfig {
    max_ingredients: number;
    max_total_actives_concentration: number;
    target_therapeutic_vectors: string[];
    skin_penetration_requirements: string[];
    stability_requirements: string[];
    cost_constraints: {min: number; max: number};
    regulatory_regions: string[];
}

export interface TherapeuticVector {
    name: string;
    target_skin_layers: string[];
    mechanism: string;
    required_concentration_range: {min: number; max: number};
    synergy_potentials: Map<string, number>;
    efficacy_evidence_level: 'theoretical' | 'in_vitro' | 'in_vivo' | 'clinical';
}

export interface MultiscaleOptimizationResult {
    reduced_search_space: CosmeticIngredient[];
    estimated_concentrations: Map<string, number>;
    therapeutic_vector_coverage: Map<string, number>;
    regulatory_compliance_score: number;
    synergy_matrix: Map<string, Map<string, number>>;
    optimization_metrics: {
        space_reduction_ratio: number;
        constraint_satisfaction_score: number;
        predicted_efficacy_score: number;
        safety_score: number;
    };
}

export class INCISearchSpaceReducer {
    private ingredientDatabase: Map<string, CosmeticIngredient>;
    private therapeuticVectors: Map<string, TherapeuticVector>;
    private regulatoryConstraints: Map<string, INCIConstraint>;
    private synergyCombinations: Map<string, Set<string>>;
    private incompatibilityMatrix: Map<string, Set<string>>;

    constructor() {
        this.ingredientDatabase = new Map();
        this.therapeuticVectors = new Map();
        this.regulatoryConstraints = new Map();
        this.synergyCombinations = new Map();
        this.incompatibilityMatrix = new Map();
        
        this.initializeDatabase();
    }

    /**
     * Initialize the ingredient database with comprehensive cosmetic chemistry data
     */
    private initializeDatabase(): void {
        // Initialize therapeutic vectors for multiscale optimization
        this.initializeTherapeuticVectors();
        
        // Load regulatory constraints for different regions
        this.initializeRegulatoryConstraints();
        
        // Initialize ingredient interaction matrices
        this.initializeInteractionMatrices();
        
        // Load comprehensive ingredient database
        this.loadIngredientDatabase();
    }

    /**
     * Initialize therapeutic vectors for different skin improvement goals
     */
    private initializeTherapeuticVectors(): void {
        const vectors: TherapeuticVector[] = [
            {
                name: 'anti_aging',
                target_skin_layers: ['epidermis', 'dermis'],
                mechanism: 'collagen_synthesis_stimulation',
                required_concentration_range: {min: 0.1, max: 5.0},
                synergy_potentials: new Map([
                    ['peptides', 0.8],
                    ['vitamin_c', 0.7],
                    ['retinoids', 0.9]
                ]),
                efficacy_evidence_level: 'clinical'
            },
            {
                name: 'barrier_enhancement',
                target_skin_layers: ['stratum_corneum'],
                mechanism: 'lipid_bilayer_reinforcement',
                required_concentration_range: {min: 1.0, max: 15.0},
                synergy_potentials: new Map([
                    ['ceramides', 0.9],
                    ['cholesterol', 0.7],
                    ['fatty_acids', 0.8]
                ]),
                efficacy_evidence_level: 'in_vivo'
            },
            {
                name: 'hydration',
                target_skin_layers: ['stratum_corneum', 'epidermis'],
                mechanism: 'water_retention_enhancement',
                required_concentration_range: {min: 2.0, max: 20.0},
                synergy_potentials: new Map([
                    ['hyaluronic_acid', 0.9],
                    ['glycerin', 0.8],
                    ['urea', 0.7]
                ]),
                efficacy_evidence_level: 'clinical'
            },
            {
                name: 'pigmentation_control',
                target_skin_layers: ['epidermis'],
                mechanism: 'tyrosinase_inhibition',
                required_concentration_range: {min: 0.5, max: 10.0},
                synergy_potentials: new Map([
                    ['niacinamide', 0.8],
                    ['arbutin', 0.7],
                    ['vitamin_c', 0.6]
                ]),
                efficacy_evidence_level: 'clinical'
            }
        ];

        vectors.forEach(vector => {
            this.therapeuticVectors.set(vector.name, vector);
        });
    }

    /**
     * Initialize regulatory constraints for different jurisdictions
     */
    private initializeRegulatoryConstraints(): void {
        // EU Cosmetic Regulation constraints
        const euConstraints: INCIConstraint = {
            approved_ingredients: new Set([
                'aqua', 'glycerin', 'hyaluronic_acid', 'niacinamide', 'retinol',
                'vitamin_c', 'peptides', 'ceramides', 'squalane', 'jojoba_oil',
                'phenoxyethanol', 'ethylhexylglycerin', 'tocopherol'
            ]),
            concentration_limits: new Map([
                ['retinol', {min: 0.0, max: 0.3}],
                ['niacinamide', {min: 0.0, max: 12.0}],
                ['vitamin_c', {min: 0.0, max: 20.0}],
                ['phenoxyethanol', {min: 0.0, max: 1.0}]
            ]),
            incompatible_pairs: new Set([
                'retinol:vitamin_c', 'benzoyl_peroxide:retinol',
                'aha:bha', 'peptides:strong_acids'
            ]),
            regulatory_region: 'EU',
            allergen_threshold: 0.001
        };

        // FDA constraints (more permissive in some areas)
        const fdaConstraints: INCIConstraint = {
            approved_ingredients: new Set([
                'water', 'glycerin', 'hyaluronic_acid', 'niacinamide', 'retinol',
                'l_ascorbic_acid', 'peptides', 'ceramides', 'squalane', 'jojoba_oil',
                'phenoxyethanol', 'ethylhexylglycerin', 'tocopherol', 'hydroquinone'
            ]),
            concentration_limits: new Map([
                ['retinol', {min: 0.0, max: 1.0}],
                ['niacinamide', {min: 0.0, max: 10.0}],
                ['l_ascorbic_acid', {min: 0.0, max: 20.0}],
                ['hydroquinone', {min: 0.0, max: 2.0}]
            ]),
            incompatible_pairs: new Set([
                'retinol:l_ascorbic_acid', 'hydroquinone:retinol'
            ]),
            regulatory_region: 'FDA',
            allergen_threshold: 0.01
        };

        this.regulatoryConstraints.set('EU', euConstraints);
        this.regulatoryConstraints.set('FDA', fdaConstraints);
    }

    /**
     * Initialize ingredient interaction matrices for synergy and incompatibility
     */
    private initializeInteractionMatrices(): void {
        // Synergistic combinations
        const synergyData = [
            ['vitamin_c', 'vitamin_e'],
            ['niacinamide', 'hyaluronic_acid'],
            ['peptides', 'vitamin_c'],
            ['ceramides', 'cholesterol'],
            ['retinol', 'peptides']
        ];

        synergyData.forEach(([ing1, ing2]) => {
            if (!this.synergyCombinations.has(ing1)) {
                this.synergyCombinations.set(ing1, new Set());
            }
            if (!this.synergyCombinations.has(ing2)) {
                this.synergyCombinations.set(ing2, new Set());
            }
            this.synergyCombinations.get(ing1)!.add(ing2);
            this.synergyCombinations.get(ing2)!.add(ing1);
        });

        // Incompatible combinations
        const incompatibilityData = [
            ['retinol', 'vitamin_c'],
            ['benzoyl_peroxide', 'retinol'],
            ['aha', 'retinol'],
            ['copper_peptides', 'vitamin_c']
        ];

        incompatibilityData.forEach(([ing1, ing2]) => {
            if (!this.incompatibilityMatrix.has(ing1)) {
                this.incompatibilityMatrix.set(ing1, new Set());
            }
            if (!this.incompatibilityMatrix.has(ing2)) {
                this.incompatibilityMatrix.set(ing2, new Set());
            }
            this.incompatibilityMatrix.get(ing1)!.add(ing2);
            this.incompatibilityMatrix.get(ing2)!.add(ing1);
        });
    }

    /**
     * Load comprehensive ingredient database
     */
    private loadIngredientDatabase(): void {
        const ingredients: CosmeticIngredient[] = [
            {
                id: 'hyaluronic_acid',
                name: 'Hyaluronic Acid',
                inci_name: 'Sodium Hyaluronate',
                category: 'ACTIVE_INGREDIENT' as IngredientCategory,
                subtype: 'HUMECTANT',
                functions: ['moisturizing', 'plumping', 'barrier_enhancement'],
                molecularWeight: 1000000,
                solubility: 'water_soluble',
                ph_stability_range: {min: 3.0, max: 8.0},
                concentration_range: {min: 0.1, max: 2.0},
                allergenicity: 'very_low',
                pregnancy_safe: true,
                therapeutic_vectors: ['hydration', 'barrier_enhancement'],
                skin_penetration_depth: 'stratum_corneum',
                onset_time_hours: 1,
                duration_hours: 24,
                stability_factors: ['ph_sensitive', 'heat_sensitive'],
                regulatory_status: new Map([['EU', 'approved'], ['FDA', 'approved']]),
                evidence_level: 'clinical',
                cost_per_gram: 0.15
            },
            {
                id: 'niacinamide',
                name: 'Niacinamide',
                inci_name: 'Niacinamide',
                category: 'ACTIVE_INGREDIENT' as IngredientCategory,
                subtype: 'VITAMIN',
                functions: ['brightening', 'pore_minimizing', 'oil_control', 'barrier_enhancement'],
                molecularWeight: 122.12,
                solubility: 'water_soluble',
                ph_stability_range: {min: 5.0, max: 7.0},
                concentration_range: {min: 2.0, max: 12.0},
                allergenicity: 'very_low',
                pregnancy_safe: true,
                therapeutic_vectors: ['pigmentation_control', 'barrier_enhancement'],
                skin_penetration_depth: 'epidermis',
                onset_time_hours: 4,
                duration_hours: 12,
                stability_factors: ['light_stable', 'heat_stable'],
                regulatory_status: new Map([['EU', 'approved'], ['FDA', 'approved']]),
                evidence_level: 'clinical',
                cost_per_gram: 0.08
            },
            {
                id: 'retinol',
                name: 'Retinol',
                inci_name: 'Retinol',
                category: 'ACTIVE_INGREDIENT' as IngredientCategory,
                subtype: 'VITAMIN',
                functions: ['anti_aging', 'cell_turnover', 'collagen_synthesis'],
                molecularWeight: 286.45,
                solubility: 'oil_soluble',
                ph_stability_range: {min: 5.5, max: 7.0},
                concentration_range: {min: 0.01, max: 1.0},
                allergenicity: 'medium',
                pregnancy_safe: false,
                therapeutic_vectors: ['anti_aging'],
                skin_penetration_depth: 'dermis',
                onset_time_hours: 72,
                duration_hours: 168,
                stability_factors: ['light_sensitive', 'oxygen_sensitive', 'heat_sensitive'],
                regulatory_status: new Map([['EU', 'approved'], ['FDA', 'approved']]),
                evidence_level: 'clinical',
                cost_per_gram: 2.50
            }
        ];

        ingredients.forEach(ingredient => {
            this.ingredientDatabase.set(ingredient.id, ingredient);
        });
    }

    /**
     * Main search space reduction algorithm with multiscale optimization
     */
    public async reduceSearchSpace(
        targetFormulation: Partial<CosmeticFormulation>,
        config: SearchSpaceReductionConfig
    ): Promise<MultiscaleOptimizationResult> {
        // Step 1: Apply INCI constraints
        const inciFilteredIngredients = this.applyINCIConstraints(
            Array.from(this.ingredientDatabase.values()),
            config.regulatory_regions
        );

        // Step 2: Filter by therapeutic vector requirements
        const therapeuticFilteredIngredients = this.filterByTherapeuticVectors(
            inciFilteredIngredients,
            config.target_therapeutic_vectors
        );

        // Step 3: Apply compatibility constraints
        const compatibilityFilteredIngredients = this.applyCompatibilityConstraints(
            therapeuticFilteredIngredients,
            config.max_ingredients
        );

        // Step 4: Optimize concentrations with multiscale considerations
        const concentrationOptimization = this.optimizeConcentrations(
            compatibilityFilteredIngredients,
            config.target_therapeutic_vectors,
            config.max_total_actives_concentration
        );

        // Step 5: Calculate therapeutic vector coverage
        const therapeuticCoverage = this.calculateTherapeuticVectorCoverage(
            concentrationOptimization.ingredients,
            concentrationOptimization.concentrations,
            config.target_therapeutic_vectors
        );

        // Step 6: Generate synergy matrix
        const synergyMatrix = this.generateSynergyMatrix(
            concentrationOptimization.ingredients
        );

        // Step 7: Calculate regulatory compliance score
        const complianceScore = this.calculateRegulatoryComplianceScore(
            concentrationOptimization.ingredients,
            concentrationOptimization.concentrations,
            config.regulatory_regions
        );

        // Step 8: Calculate optimization metrics
        const optimizationMetrics = this.calculateOptimizationMetrics(
            Array.from(this.ingredientDatabase.values()),
            concentrationOptimization.ingredients,
            therapeuticCoverage,
            complianceScore,
            synergyMatrix
        );

        return {
            reduced_search_space: concentrationOptimization.ingredients,
            estimated_concentrations: concentrationOptimization.concentrations,
            therapeutic_vector_coverage: therapeuticCoverage,
            regulatory_compliance_score: complianceScore,
            synergy_matrix: synergyMatrix,
            optimization_metrics: optimizationMetrics
        };
    }

    /**
     * Apply INCI constraints to filter ingredients
     */
    private applyINCIConstraints(
        ingredients: CosmeticIngredient[],
        regions: string[]
    ): CosmeticIngredient[] {
        return ingredients.filter(ingredient => {
            return regions.every(region => {
                const constraint = this.regulatoryConstraints.get(region);
                if (!constraint) return false;
                
                return constraint.approved_ingredients.has(ingredient.id) ||
                       constraint.approved_ingredients.has(ingredient.inci_name.toLowerCase());
            });
        });
    }

    /**
     * Filter ingredients by therapeutic vector requirements
     */
    private filterByTherapeuticVectors(
        ingredients: CosmeticIngredient[],
        targetVectors: string[]
    ): CosmeticIngredient[] {
        return ingredients.filter(ingredient => {
            return targetVectors.some(vector => 
                ingredient.therapeutic_vectors?.includes(vector)
            );
        });
    }

    /**
     * Apply compatibility constraints using graph-based analysis
     */
    private applyCompatibilityConstraints(
        ingredients: CosmeticIngredient[],
        maxIngredients: number
    ): CosmeticIngredient[] {
        // Create compatibility graph
        const compatibilityGraph = new Map<string, Set<string>>();
        
        ingredients.forEach(ing => {
            compatibilityGraph.set(ing.id, new Set());
        });

        // Add edges for compatible ingredients
        ingredients.forEach(ing1 => {
            ingredients.forEach(ing2 => {
                if (ing1.id !== ing2.id && this.areIngredientsCompatible(ing1.id, ing2.id)) {
                    compatibilityGraph.get(ing1.id)!.add(ing2.id);
                }
            });
        });

        // Find maximum clique (simplified greedy approach)
        return this.findMaximalCompatibleSet(ingredients, compatibilityGraph, maxIngredients);
    }

    /**
     * Check if two ingredients are compatible
     */
    private areIngredientsCompatible(ing1: string, ing2: string): boolean {
        const incompatibles1 = this.incompatibilityMatrix.get(ing1) || new Set();
        const incompatibles2 = this.incompatibilityMatrix.get(ing2) || new Set();
        
        return !incompatibles1.has(ing2) && !incompatibles2.has(ing1);
    }

    /**
     * Find maximal compatible set of ingredients
     */
    private findMaximalCompatibleSet(
        ingredients: CosmeticIngredient[],
        compatibilityGraph: Map<string, Set<string>>,
        maxSize: number
    ): CosmeticIngredient[] {
        // Greedy algorithm: start with ingredient with most connections
        const sorted = ingredients.sort((a, b) => 
            (compatibilityGraph.get(b.id)?.size || 0) - (compatibilityGraph.get(a.id)?.size || 0)
        );

        const selected = new Set<string>();
        const result: CosmeticIngredient[] = [];

        for (const ingredient of sorted) {
            if (result.length >= maxSize) break;
            
            const isCompatibleWithSelected = Array.from(selected).every(selectedId =>
                this.areIngredientsCompatible(ingredient.id, selectedId)
            );

            if (isCompatibleWithSelected) {
                selected.add(ingredient.id);
                result.push(ingredient);
            }
        }

        return result;
    }

    /**
     * Optimize ingredient concentrations using multiscale constraints
     */
    private optimizeConcentrations(
        ingredients: CosmeticIngredient[],
        targetVectors: string[],
        maxTotalActives: number
    ): {ingredients: CosmeticIngredient[]; concentrations: Map<string, number>} {
        const concentrations = new Map<string, number>();
        let totalActives = 0;

        // Sort ingredients by effectiveness for target vectors
        const sortedIngredients = ingredients.sort((a, b) => {
            const scoreA = this.calculateIngredientEffectivenessScore(a, targetVectors);
            const scoreB = this.calculateIngredientEffectivenessScore(b, targetVectors);
            return scoreB - scoreA;
        });

        // Allocate concentrations based on effectiveness and constraints
        for (const ingredient of sortedIngredients) {
            const minConc = ingredient.concentration_range?.min || 0.1;
            const maxConc = Math.min(
                ingredient.concentration_range?.max || 10.0,
                maxTotalActives - totalActives
            );

            if (maxConc >= minConc && totalActives < maxTotalActives) {
                // Calculate optimal concentration based on therapeutic vectors
                const optimalConc = this.calculateOptimalConcentration(
                    ingredient,
                    targetVectors,
                    minConc,
                    maxConc
                );

                concentrations.set(ingredient.id, optimalConc);
                totalActives += optimalConc;
            }
        }

        // Filter ingredients that received concentration allocations
        const finalIngredients = sortedIngredients.filter(ing => 
            concentrations.has(ing.id)
        );

        return {ingredients: finalIngredients, concentrations};
    }

    /**
     * Calculate ingredient effectiveness score for target therapeutic vectors
     */
    private calculateIngredientEffectivenessScore(
        ingredient: CosmeticIngredient,
        targetVectors: string[]
    ): number {
        let score = 0;
        
        targetVectors.forEach(vector => {
            if (ingredient.therapeutic_vectors?.includes(vector)) {
                const vectorData = this.therapeuticVectors.get(vector);
                if (vectorData) {
                    // Base score from evidence level
                    const evidenceScore = this.getEvidenceScore(ingredient.evidence_level || 'theoretical');
                    
                    // Synergy bonus
                    const synergyBonus = vectorData.synergy_potentials.get(ingredient.id) || 0;
                    
                    // Safety factor
                    const safetyFactor = this.getSafetyFactor(ingredient.allergenicity);
                    
                    score += evidenceScore * (1 + synergyBonus) * safetyFactor;
                }
            }
        });

        return score;
    }

    /**
     * Get evidence score based on evidence level
     */
    private getEvidenceScore(evidenceLevel: string): number {
        const scores = {
            'clinical': 1.0,
            'in_vivo': 0.8,
            'in_vitro': 0.6,
            'theoretical': 0.4
        };
        return scores[evidenceLevel as keyof typeof scores] || 0.4;
    }

    /**
     * Get safety factor based on allergenicity
     */
    private getSafetyFactor(allergenicity: string): number {
        const factors = {
            'very_low': 1.0,
            'low': 0.9,
            'medium': 0.7,
            'high': 0.5
        };
        return factors[allergenicity as keyof typeof factors] || 0.5;
    }

    /**
     * Calculate optimal concentration for an ingredient
     */
    private calculateOptimalConcentration(
        ingredient: CosmeticIngredient,
        targetVectors: string[],
        minConc: number,
        maxConc: number
    ): number {
        // Start with middle of range
        let optimalConc = (minConc + maxConc) / 2;

        // Adjust based on therapeutic vector requirements
        targetVectors.forEach(vector => {
            const vectorData = this.therapeuticVectors.get(vector);
            if (vectorData && ingredient.therapeutic_vectors?.includes(vector)) {
                const vectorOptimal = (vectorData.required_concentration_range.min + 
                                    vectorData.required_concentration_range.max) / 2;
                
                // Weight the optimal concentration towards vector requirements
                optimalConc = (optimalConc + vectorOptimal) / 2;
            }
        });

        return Math.max(minConc, Math.min(maxConc, optimalConc));
    }

    /**
     * Calculate therapeutic vector coverage
     */
    private calculateTherapeuticVectorCoverage(
        ingredients: CosmeticIngredient[],
        concentrations: Map<string, number>,
        targetVectors: string[]
    ): Map<string, number> {
        const coverage = new Map<string, number>();

        targetVectors.forEach(vector => {
            let vectorCoverage = 0;
            const vectorData = this.therapeuticVectors.get(vector);

            if (vectorData) {
                ingredients.forEach(ingredient => {
                    if (ingredient.therapeutic_vectors?.includes(vector)) {
                        const concentration = concentrations.get(ingredient.id) || 0;
                        const effectiveness = this.calculateIngredientEffectivenessScore(
                            ingredient, [vector]
                        );
                        
                        // Normalize concentration to 0-1 range
                        const normalizedConc = concentration / (vectorData.required_concentration_range.max);
                        vectorCoverage += effectiveness * normalizedConc;
                    }
                });
            }

            coverage.set(vector, Math.min(1.0, vectorCoverage));
        });

        return coverage;
    }

    /**
     * Generate synergy matrix for selected ingredients
     */
    private generateSynergyMatrix(ingredients: CosmeticIngredient[]): Map<string, Map<string, number>> {
        const matrix = new Map<string, Map<string, number>>();

        ingredients.forEach(ing1 => {
            const row = new Map<string, number>();
            
            ingredients.forEach(ing2 => {
                if (ing1.id !== ing2.id) {
                    const synergyScore = this.calculateSynergyScore(ing1.id, ing2.id);
                    row.set(ing2.id, synergyScore);
                }
            });
            
            matrix.set(ing1.id, row);
        });

        return matrix;
    }

    /**
     * Calculate synergy score between two ingredients
     */
    private calculateSynergyScore(ing1: string, ing2: string): number {
        const synergies1 = this.synergyCombinations.get(ing1) || new Set();
        const synergies2 = this.synergyCombinations.get(ing2) || new Set();

        if (synergies1.has(ing2) || synergies2.has(ing1)) {
            return 0.8; // High synergy
        }

        // Check for indirect synergies through shared therapeutic vectors
        const ingredient1 = this.ingredientDatabase.get(ing1);
        const ingredient2 = this.ingredientDatabase.get(ing2);

        if (ingredient1 && ingredient2) {
            const sharedVectors = ingredient1.therapeutic_vectors?.filter(v =>
                ingredient2.therapeutic_vectors?.includes(v)
            ) || [];

            if (sharedVectors.length > 0) {
                return 0.4; // Moderate synergy
            }
        }

        return 0.1; // Minimal synergy
    }

    /**
     * Calculate regulatory compliance score
     */
    private calculateRegulatoryComplianceScore(
        ingredients: CosmeticIngredient[],
        concentrations: Map<string, number>,
        regions: string[]
    ): number {
        let totalScore = 0;
        let totalChecks = 0;

        regions.forEach(region => {
            const constraint = this.regulatoryConstraints.get(region);
            if (!constraint) return;

            ingredients.forEach(ingredient => {
                totalChecks++;

                // Check if ingredient is approved
                if (!constraint.approved_ingredients.has(ingredient.id)) {
                    return; // Score remains 0 for this check
                }

                // Check concentration limits
                const concentration = concentrations.get(ingredient.id) || 0;
                const limits = constraint.concentration_limits.get(ingredient.id);

                if (limits) {
                    if (concentration >= limits.min && concentration <= limits.max) {
                        totalScore++;
                    }
                } else {
                    // No specific limits, assume compliance
                    totalScore++;
                }
            });
        });

        return totalChecks > 0 ? totalScore / totalChecks : 0;
    }

    /**
     * Calculate comprehensive optimization metrics
     */
    private calculateOptimizationMetrics(
        originalIngredients: CosmeticIngredient[],
        optimizedIngredients: CosmeticIngredient[],
        therapeuticCoverage: Map<string, number>,
        complianceScore: number,
        synergyMatrix: Map<string, Map<string, number>>
    ): MultiscaleOptimizationResult['optimization_metrics'] {
        // Space reduction ratio
        const spaceReductionRatio = 1 - (optimizedIngredients.length / originalIngredients.length);

        // Constraint satisfaction score (average of compliance and coverage)
        const avgCoverage = Array.from(therapeuticCoverage.values()).reduce((a, b) => a + b, 0) / 
                           therapeuticCoverage.size;
        const constraintSatisfactionScore = (complianceScore + avgCoverage) / 2;

        // Predicted efficacy score based on ingredient quality and synergies
        const efficacyScore = this.calculateEfficacyScore(optimizedIngredients, synergyMatrix);

        // Safety score based on allergenicity and pregnancy safety
        const safetyScore = this.calculateSafetyScore(optimizedIngredients);

        return {
            space_reduction_ratio: spaceReductionRatio,
            constraint_satisfaction_score: constraintSatisfactionScore,
            predicted_efficacy_score: efficacyScore,
            safety_score: safetyScore
        };
    }

    /**
     * Calculate predicted efficacy score
     */
    private calculateEfficacyScore(
        ingredients: CosmeticIngredient[],
        synergyMatrix: Map<string, Map<string, number>>
    ): number {
        let baseEfficacy = 0;
        let synergyBonus = 0;

        // Base efficacy from individual ingredients
        ingredients.forEach(ingredient => {
            baseEfficacy += this.getEvidenceScore(ingredient.evidence_level || 'theoretical');
        });

        // Synergy bonus
        ingredients.forEach(ing1 => {
            ingredients.forEach(ing2 => {
                if (ing1.id !== ing2.id) {
                    const synergy = synergyMatrix.get(ing1.id)?.get(ing2.id) || 0;
                    synergyBonus += synergy;
                }
            });
        });

        // Normalize scores
        const normalizedBase = ingredients.length > 0 ? baseEfficacy / ingredients.length : 0;
        const normalizedSynergy = ingredients.length > 1 ? 
            synergyBonus / (ingredients.length * (ingredients.length - 1)) : 0;

        return (normalizedBase + normalizedSynergy * 0.3) / 1.3;
    }

    /**
     * Calculate safety score
     */
    private calculateSafetyScore(ingredients: CosmeticIngredient[]): number {
        let totalSafety = 0;

        ingredients.forEach(ingredient => {
            let ingredientSafety = this.getSafetyFactor(ingredient.allergenicity);
            
            // Pregnancy safety bonus
            if (ingredient.pregnancy_safe) {
                ingredientSafety *= 1.1;
            }

            totalSafety += ingredientSafety;
        });

        return ingredients.length > 0 ? totalSafety / ingredients.length : 0;
    }
}

/**
 * Utility functions for INCI processing and validation
 */
export class INCIUtilities {
    /**
     * Parse INCI list from product labeling
     */
    static parseINCIList(inciString: string): string[] {
        return inciString
            .toLowerCase()
            .split(/[,;]/)
            .map(ingredient => ingredient.trim())
            .filter(ingredient => ingredient.length > 0);
    }

    /**
     * Validate INCI ordering based on concentration rules
     */
    static validateINCIOrdering(
        inciList: string[],
        concentrations: Map<string, number>
    ): boolean {
        for (let i = 0; i < inciList.length - 1; i++) {
            const currentConc = concentrations.get(inciList[i]) || 0;
            const nextConc = concentrations.get(inciList[i + 1]) || 0;
            
            if (currentConc < nextConc) {
                return false; // Concentrations should be in descending order
            }
        }
        return true;
    }

    /**
     * Estimate concentrations from INCI ordering
     */
    static estimateConcentrationsFromOrdering(
        inciList: string[],
        totalConcentration: number = 100
    ): Map<string, number> {
        const concentrations = new Map<string, number>();
        
        // Use Zipf's law approximation for concentration distribution
        let totalWeight = 0;
        for (let i = 1; i <= inciList.length; i++) {
            totalWeight += 1 / i;
        }

        inciList.forEach((ingredient, index) => {
            const weight = 1 / (index + 1);
            const concentration = (weight / totalWeight) * totalConcentration;
            concentrations.set(ingredient, concentration);
        });

        return concentrations;
    }
}