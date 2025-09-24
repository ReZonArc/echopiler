/**
 * Adaptive Attention Allocation for Cosmeceutical Formulation Optimization
 *
 * This module implements an attention allocation system inspired by OpenCog's ECAN
 * (Economic Attention Networks) for efficiently managing computational resources
 * during cosmeceutical formulation optimization. The system dynamically allocates
 * attention to promising ingredient combinations, regulatory constraints, and
 * market opportunities.
 *
 * Key Features:
 * - Short-term Importance (STI) for immediate formulation needs
 * - Long-term Importance (LTI) for strategic ingredient development
 * - Dynamic attention decay and reinforcement mechanisms
 * - Market opportunity detection and resource allocation
 * - Regulatory compliance prioritization
 */

import {CosmeticIngredient} from '../../types/cheminformatics/cosmetic-chemistry.interfaces.js';

export interface AttentionAtom {
    id: string;
    type: 'ingredient' | 'combination' | 'formulation' | 'constraint' | 'market_opportunity';
    content: any; // The actual ingredient, combination, etc.
    short_term_importance: number; // STI: 0-1000
    long_term_importance: number; // LTI: 0-1000
    very_long_term_importance: number; // VLTI: 0-1000
    attention_value: number; // AV: computed from STI and LTI
    last_accessed: Date;
    access_count: number;
    creation_time: Date;
    confidence: number; // 0-1
    utility: number; // 0-1
    cost: number; // computational cost
    market_relevance: number; // 0-1
    regulatory_risk: number; // 0-1
}

export interface AttentionAllocationConfig {
    max_attention_atoms: number;
    sti_decay_rate: number; // per time unit
    lti_decay_rate: number; // per time unit
    vlti_decay_rate: number; // per time unit
    attention_threshold: number; // minimum AV to maintain
    reinforcement_factor: number; // reward multiplier for successful predictions
    exploration_factor: number; // randomness in selection
    cost_penalty_factor: number; // penalty for high computational cost
    market_weight: number; // importance of market relevance
    regulatory_weight: number; // importance of regulatory compliance
}

export interface AttentionDistribution {
    high_attention: AttentionAtom[];
    medium_attention: AttentionAtom[];
    low_attention: AttentionAtom[];
    focus_areas: string[];
    resource_allocation: Map<string, number>;
    next_computation_targets: AttentionAtom[];
}

export interface MarketOpportunity {
    id: string;
    trend_name: string;
    ingredient_gaps: string[];
    market_size_estimate: number;
    growth_rate: number;
    competitive_landscape: string;
    regulatory_barriers: string[];
    time_to_market: number;
    confidence_level: number;
}

export class AdaptiveAttentionAllocator {
    private attentionSpace: Map<string, AttentionAtom>;
    private config: AttentionAllocationConfig;
    private marketOpportunities: Map<string, MarketOpportunity>;
    private regulatoryPriorities: Map<string, number>;
    private computationHistory: Map<string, {success: boolean; cost: number; time: Date}[]>;

    constructor(config?: Partial<AttentionAllocationConfig>) {
        this.attentionSpace = new Map();
        this.marketOpportunities = new Map();
        this.regulatoryPriorities = new Map();
        this.computationHistory = new Map();

        this.config = {
            max_attention_atoms: 1000,
            sti_decay_rate: 0.1,
            lti_decay_rate: 0.01,
            vlti_decay_rate: 0.001,
            attention_threshold: 10,
            reinforcement_factor: 1.5,
            exploration_factor: 0.1,
            cost_penalty_factor: 0.2,
            market_weight: 0.3,
            regulatory_weight: 0.4,
            ...config,
        };

        this.initializeMarketOpportunities();
        this.initializeRegulatoryPriorities();
    }

    /**
     * Initialize market opportunities for attention allocation
     */
    private initializeMarketOpportunities(): void {
        const opportunities: MarketOpportunity[] = [
            {
                id: 'sustainable_packaging',
                trend_name: 'Sustainable Beauty Packaging',
                ingredient_gaps: ['biodegradable_polymers', 'plant_based_preservatives'],
                market_size_estimate: 2.3e9, // $2.3B
                growth_rate: 0.15, // 15% annually
                competitive_landscape: 'emerging',
                regulatory_barriers: ['biodegradability_certification', 'packaging_directives'],
                time_to_market: 18, // months
                confidence_level: 0.8,
            },
            {
                id: 'personalized_skincare',
                trend_name: 'AI-Driven Personalized Skincare',
                ingredient_gaps: ['adaptive_actives', 'biomarker_responsive_ingredients'],
                market_size_estimate: 4.1e9, // $4.1B
                growth_rate: 0.22, // 22% annually
                competitive_landscape: 'competitive',
                regulatory_barriers: ['personalized_medicine_regulations', 'data_privacy'],
                time_to_market: 24, // months
                confidence_level: 0.7,
            },
            {
                id: 'microbiome_beauty',
                trend_name: 'Microbiome-Friendly Cosmetics',
                ingredient_gaps: ['prebiotic_actives', 'microbiome_modulators'],
                market_size_estimate: 1.8e9, // $1.8B
                growth_rate: 0.35, // 35% annually
                competitive_landscape: 'early_stage',
                regulatory_barriers: ['probiotic_regulations', 'safety_assessments'],
                time_to_market: 30, // months
                confidence_level: 0.6,
            },
            {
                id: 'clean_beauty',
                trend_name: 'Clean Beauty Movement',
                ingredient_gaps: ['natural_preservatives', 'plant_based_actives'],
                market_size_estimate: 5.7e9, // $5.7B
                growth_rate: 0.18, // 18% annually
                competitive_landscape: 'saturated',
                regulatory_barriers: ['natural_certification', 'ingredient_restrictions'],
                time_to_market: 12, // months
                confidence_level: 0.9,
            },
        ];

        opportunities.forEach(opp => {
            this.marketOpportunities.set(opp.id, opp);
        });
    }

    /**
     * Initialize regulatory priorities for different regions
     */
    private initializeRegulatoryPriorities(): void {
        // Higher values = higher priority for compliance
        this.regulatoryPriorities.set('EU_REACH', 0.9);
        this.regulatoryPriorities.set('EU_Cosmetic_Regulation', 0.95);
        this.regulatoryPriorities.set('FDA_GRAS', 0.8);
        this.regulatoryPriorities.set('Health_Canada', 0.75);
        this.regulatoryPriorities.set('ASEAN_Cosmetic_Directive', 0.7);
        this.regulatoryPriorities.set('China_NMPA', 0.85);
        this.regulatoryPriorities.set('Japan_MHLW', 0.8);
    }

    /**
     * Add or update an attention atom in the attention space
     */
    public addAttentionAtom(atom: Partial<AttentionAtom>): void {
        const now = new Date();
        const existingAtom = this.attentionSpace.get(atom.id!);

        const newAtom: AttentionAtom = {
            id: atom.id!,
            type: atom.type || 'ingredient',
            content: atom.content,
            short_term_importance: atom.short_term_importance || 100,
            long_term_importance: atom.long_term_importance || 100,
            very_long_term_importance: atom.very_long_term_importance || 100,
            attention_value: 0, // Will be computed
            last_accessed: now,
            access_count: existingAtom?.access_count || 0,
            creation_time: existingAtom?.creation_time || now,
            confidence: atom.confidence || 0.5,
            utility: atom.utility || 0.5,
            cost: atom.cost || 1.0,
            market_relevance: atom.market_relevance || 0.5,
            regulatory_risk: atom.regulatory_risk || 0.5,
            ...atom,
        };

        newAtom.attention_value = this.computeAttentionValue(newAtom);
        this.attentionSpace.set(newAtom.id, newAtom);

        // Garbage collection if we exceed maximum atoms
        if (this.attentionSpace.size > this.config.max_attention_atoms) {
            this.performAttentionGarbageCollection();
        }
    }

    /**
     * Compute attention value from STI, LTI, and other factors
     */
    private computeAttentionValue(atom: AttentionAtom): number {
        const timeFactors = this.computeTimeFactors(atom);
        const marketFactor = atom.market_relevance * this.config.market_weight;
        const regulatoryFactor = (1 - atom.regulatory_risk) * this.config.regulatory_weight;
        const costPenalty = atom.cost * this.config.cost_penalty_factor;

        const baseAV =
            (atom.short_term_importance * timeFactors.sti_factor +
                atom.long_term_importance * timeFactors.lti_factor +
                atom.very_long_term_importance * timeFactors.vlti_factor) /
            3;

        const adjustedAV =
            baseAV * (1 + marketFactor + regulatoryFactor) * atom.confidence * atom.utility * (1 - costPenalty);

        return Math.max(0, Math.min(1000, adjustedAV));
    }

    /**
     * Compute time-based decay factors
     */
    private computeTimeFactors(atom: AttentionAtom): {sti_factor: number; lti_factor: number; vlti_factor: number} {
        const now = new Date();
        const timeSinceAccess = (now.getTime() - atom.last_accessed.getTime()) / (1000 * 60 * 60); // hours
        const timeSinceCreation = (now.getTime() - atom.creation_time.getTime()) / (1000 * 60 * 60 * 24); // days

        const sti_factor = Math.exp(-this.config.sti_decay_rate * timeSinceAccess);
        const lti_factor = Math.exp(-this.config.lti_decay_rate * timeSinceCreation);
        const vlti_factor = Math.exp(-this.config.vlti_decay_rate * timeSinceCreation);

        return {sti_factor, lti_factor, vlti_factor};
    }

    /**
     * Update attention values for all atoms based on decay
     */
    public updateAttentionDecay(): void {
        this.attentionSpace.forEach((atom, id) => {
            atom.attention_value = this.computeAttentionValue(atom);

            // Remove atoms below attention threshold
            if (atom.attention_value < this.config.attention_threshold) {
                this.attentionSpace.delete(id);
            }
        });
    }

    /**
     * Reinforce attention for successful computations
     */
    public reinforceAttention(atomId: string, success: boolean, computationCost: number): void {
        const atom = this.attentionSpace.get(atomId);
        if (!atom) return;

        // Record computation history
        if (!this.computationHistory.has(atomId)) {
            this.computationHistory.set(atomId, []);
        }

        this.computationHistory.get(atomId)!.push({
            success,
            cost: computationCost,
            time: new Date(),
        });

        // Update importance values based on success
        if (success) {
            atom.short_term_importance = Math.min(1000, atom.short_term_importance * this.config.reinforcement_factor);
            atom.confidence = Math.min(1.0, atom.confidence * 1.1);
            atom.utility = Math.min(1.0, atom.utility * 1.05);
        } else {
            atom.short_term_importance = Math.max(1, atom.short_term_importance * 0.8);
            atom.confidence = Math.max(0.1, atom.confidence * 0.95);
        }

        // Update access information
        atom.last_accessed = new Date();
        atom.access_count++;

        // Recompute attention value
        atom.attention_value = this.computeAttentionValue(atom);
    }

    /**
     * Allocate computational attention based on current priorities
     */
    public allocateAttention(): AttentionDistribution {
        this.updateAttentionDecay();

        const sortedAtoms = Array.from(this.attentionSpace.values()).sort(
            (a, b) => b.attention_value - a.attention_value,
        );

        const totalAttention = sortedAtoms.reduce((sum, atom) => sum + atom.attention_value, 0);

        // Categorize atoms by attention level
        const highThreshold = totalAttention * 0.1; // Top 10%
        const mediumThreshold = totalAttention * 0.3; // Next 20%

        const high: AttentionAtom[] = [];
        const medium: AttentionAtom[] = [];
        const low: AttentionAtom[] = [];

        let runningSum = 0;

        sortedAtoms.forEach(atom => {
            runningSum += atom.attention_value;

            if (runningSum <= highThreshold) {
                high.push(atom);
            } else if (runningSum <= mediumThreshold) {
                medium.push(atom);
            } else {
                low.push(atom);
            }
        });

        // Identify focus areas
        const focusAreas = this.identifyFocusAreas(high);

        // Compute resource allocation
        const resourceAllocation = this.computeResourceAllocation(high, medium, low);

        // Select next computation targets
        const nextTargets = this.selectNextComputationTargets(high, medium);

        return {
            high_attention: high,
            medium_attention: medium,
            low_attention: low,
            focus_areas: focusAreas,
            resource_allocation: resourceAllocation,
            next_computation_targets: nextTargets,
        };
    }

    /**
     * Identify key focus areas from high-attention atoms
     */
    private identifyFocusAreas(highAttentionAtoms: AttentionAtom[]): string[] {
        const focusMap = new Map<string, number>();

        highAttentionAtoms.forEach(atom => {
            let focusArea: string;

            switch (atom.type) {
                case 'ingredient':
                    focusArea = `ingredient_optimization`;
                    break;
                case 'combination':
                    focusArea = `synergy_exploration`;
                    break;
                case 'formulation':
                    focusArea = `formulation_refinement`;
                    break;
                case 'constraint':
                    focusArea = `regulatory_compliance`;
                    break;
                case 'market_opportunity':
                    focusArea = `market_innovation`;
                    break;
                default:
                    focusArea = 'general_optimization';
            }

            focusMap.set(focusArea, (focusMap.get(focusArea) || 0) + atom.attention_value);
        });

        // Return top focus areas
        return Array.from(focusMap.entries())
            .sort((a, b) => b[1] - a[1])
            .slice(0, 5)
            .map(([area, _]) => area);
    }

    /**
     * Compute resource allocation percentages
     */
    private computeResourceAllocation(
        high: AttentionAtom[],
        medium: AttentionAtom[],
        low: AttentionAtom[],
    ): Map<string, number> {
        const allocation = new Map<string, number>();

        const totalHighAttention = high.reduce((sum, atom) => sum + atom.attention_value, 0);
        const totalMediumAttention = medium.reduce((sum, atom) => sum + atom.attention_value, 0);
        const totalLowAttention = low.reduce((sum, atom) => sum + atom.attention_value, 0);

        const totalAttention = totalHighAttention + totalMediumAttention + totalLowAttention;

        if (totalAttention > 0) {
            allocation.set('high_priority', (totalHighAttention / totalAttention) * 0.7); // 70% to high
            allocation.set('medium_priority', (totalMediumAttention / totalAttention) * 0.25); // 25% to medium
            allocation.set('low_priority', (totalLowAttention / totalAttention) * 0.05); // 5% to low
        } else {
            allocation.set('high_priority', 0.7);
            allocation.set('medium_priority', 0.25);
            allocation.set('low_priority', 0.05);
        }

        // Add specific allocations for different types of computation
        allocation.set('ingredient_analysis', 0.3);
        allocation.set('synergy_computation', 0.25);
        allocation.set('regulatory_checking', 0.2);
        allocation.set('market_analysis', 0.15);
        allocation.set('formulation_optimization', 0.1);

        return allocation;
    }

    /**
     * Select next computation targets using exploration-exploitation balance
     */
    private selectNextComputationTargets(high: AttentionAtom[], medium: AttentionAtom[]): AttentionAtom[] {
        const targets: AttentionAtom[] = [];

        // Exploitation: Select top high-attention atoms
        const exploitationCount = Math.floor(high.length * (1 - this.config.exploration_factor));
        targets.push(...high.slice(0, exploitationCount));

        // Exploration: Randomly select from medium attention atoms
        const explorationCount = Math.min(5, medium.length);
        const shuffledMedium = medium.sort(() => Math.random() - 0.5);
        targets.push(...shuffledMedium.slice(0, explorationCount));

        // Sort by computational efficiency (attention value / cost)
        return targets
            .sort((a, b) => {
                const efficiencyA = a.attention_value / a.cost;
                const efficiencyB = b.attention_value / b.cost;
                return efficiencyB - efficiencyA;
            })
            .slice(0, 10); // Return top 10 targets
    }

    /**
     * Perform garbage collection on attention space
     */
    private performAttentionGarbageCollection(): void {
        const atoms = Array.from(this.attentionSpace.values()).sort((a, b) => a.attention_value - b.attention_value);

        const removeCount = this.attentionSpace.size - this.config.max_attention_atoms;

        for (let i = 0; i < removeCount; i++) {
            this.attentionSpace.delete(atoms[i].id);
        }
    }

    /**
     * Process market opportunities and update attention allocation
     */
    public updateMarketOpportunityAttention(): void {
        this.marketOpportunities.forEach(opportunity => {
            const marketScore = this.calculateMarketOpportunityScore(opportunity);

            // Create or update attention atom for this market opportunity
            this.addAttentionAtom({
                id: `market_${opportunity.id}`,
                type: 'market_opportunity',
                content: opportunity,
                short_term_importance: marketScore * 200,
                long_term_importance: opportunity.growth_rate * 500,
                very_long_term_importance: (opportunity.market_size_estimate / 1e9) * 100,
                confidence: opportunity.confidence_level,
                utility: this.calculateMarketUtility(opportunity),
                cost: opportunity.time_to_market / 12, // Normalize to years
                market_relevance: 1.0, // Maximum for market opportunities
                regulatory_risk: opportunity.regulatory_barriers.length / 10, // Normalize
            });

            // Create attention atoms for ingredient gaps
            opportunity.ingredient_gaps.forEach(gap => {
                this.addAttentionAtom({
                    id: `gap_${gap}_${opportunity.id}`,
                    type: 'ingredient',
                    content: {ingredient_gap: gap, market_opportunity: opportunity.id},
                    short_term_importance: marketScore * 150,
                    long_term_importance: opportunity.growth_rate * 300,
                    very_long_term_importance: 50,
                    confidence: opportunity.confidence_level * 0.8,
                    utility: 0.8,
                    cost: 2.0, // Research cost for new ingredients
                    market_relevance: 0.9,
                    regulatory_risk: 0.7, // New ingredients typically have higher regulatory risk
                });
            });
        });
    }

    /**
     * Calculate market opportunity score
     */
    private calculateMarketOpportunityScore(opportunity: MarketOpportunity): number {
        const sizeScore = Math.log10(opportunity.market_size_estimate / 1e6) / 4; // Normalize to 0-1
        const growthScore = Math.min(1.0, opportunity.growth_rate * 2); // Cap at 100% growth
        const competitiveScore = this.getCompetitiveLandscapeScore(opportunity.competitive_landscape);
        const timeScore = Math.max(0, 1 - opportunity.time_to_market / 60); // Penalty for long time to market
        const confidenceScore = opportunity.confidence_level;

        return sizeScore * 0.3 + growthScore * 0.3 + competitiveScore * 0.2 + timeScore * 0.1 + confidenceScore * 0.1;
    }

    /**
     * Get competitive landscape score
     */
    private getCompetitiveLandscapeScore(landscape: string): number {
        const scores = {
            early_stage: 0.9,
            emerging: 0.7,
            competitive: 0.5,
            saturated: 0.2,
        };
        return scores[landscape as keyof typeof scores] || 0.5;
    }

    /**
     * Calculate market utility
     */
    private calculateMarketUtility(opportunity: MarketOpportunity): number {
        const riskAdjustedReturn =
            (opportunity.growth_rate * opportunity.market_size_estimate) / (opportunity.regulatory_barriers.length + 1);

        const timeAdjustedReturn = riskAdjustedReturn / (opportunity.time_to_market / 12) ** 0.5;

        // Normalize to 0-1 range
        return Math.min(1.0, timeAdjustedReturn / 1e10);
    }

    /**
     * Update regulatory compliance attention based on current priorities
     */
    public updateRegulatoryAttention(ingredients: CosmeticIngredient[]): void {
        ingredients.forEach(ingredient => {
            this.regulatoryPriorities.forEach((priority, regulation) => {
                const complianceRisk = this.assessRegulatoryRisk(ingredient, regulation);

                this.addAttentionAtom({
                    id: `regulatory_${ingredient.id}_${regulation}`,
                    type: 'constraint',
                    content: {ingredient: ingredient.id, regulation},
                    short_term_importance: complianceRisk * priority * 300,
                    long_term_importance: priority * 200,
                    very_long_term_importance: 100,
                    confidence: 0.8,
                    utility: priority,
                    cost: complianceRisk * 2,
                    market_relevance: 0.6,
                    regulatory_risk: complianceRisk,
                });
            });
        });
    }

    /**
     * Assess regulatory risk for an ingredient under specific regulation
     */
    private assessRegulatoryRisk(ingredient: CosmeticIngredient, regulation: string): number {
        let risk = 0.3; // Base risk

        // Higher risk for new/novel ingredients
        if (ingredient.evidence_level === 'theoretical') {
            risk += 0.4;
        } else if (ingredient.evidence_level === 'in_vitro') {
            risk += 0.2;
        }

        // Higher risk for high-allergenicity ingredients
        if (ingredient.allergenicity === 'high') {
            risk += 0.3;
        } else if (ingredient.allergenicity === 'medium') {
            risk += 0.1;
        }

        // Higher risk if not pregnancy safe
        if (!ingredient.pregnancy_safe) {
            risk += 0.2;
        }

        // Check regulatory status
        const status = ingredient.regulatory_status?.get(regulation.split('_')[0]);
        if (status === 'pending' || status === 'restricted') {
            risk += 0.3;
        } else if (status === 'approved') {
            risk -= 0.2;
        }

        return Math.max(0, Math.min(1, risk));
    }

    /**
     * Get attention statistics and insights
     */
    public getAttentionStatistics(): {
        total_atoms: number;
        attention_distribution: {high: number; medium: number; low: number};
        top_focus_areas: string[];
        computational_efficiency: number;
        market_opportunity_coverage: number;
        regulatory_compliance_level: number;
    } {
        const distribution = this.allocateAttention();

        const computationalEfficiency = this.calculateComputationalEfficiency();
        const marketCoverage = this.calculateMarketOpportunityCoverage();
        const complianceLevel = this.calculateRegulatoryComplianceLevel();

        return {
            total_atoms: this.attentionSpace.size,
            attention_distribution: {
                high: distribution.high_attention.length,
                medium: distribution.medium_attention.length,
                low: distribution.low_attention.length,
            },
            top_focus_areas: distribution.focus_areas,
            computational_efficiency: computationalEfficiency,
            market_opportunity_coverage: marketCoverage,
            regulatory_compliance_level: complianceLevel,
        };
    }

    /**
     * Calculate computational efficiency based on attention allocation
     */
    private calculateComputationalEfficiency(): number {
        const recentHistory = Array.from(this.computationHistory.values())
            .flat()
            .filter(entry => {
                const daysSince = (Date.now() - entry.time.getTime()) / (1000 * 60 * 60 * 24);
                return daysSince <= 7; // Last week
            });

        if (recentHistory.length === 0) return 0.5;

        const successRate = recentHistory.filter(entry => entry.success).length / recentHistory.length;
        const avgCost = recentHistory.reduce((sum, entry) => sum + entry.cost, 0) / recentHistory.length;

        // Efficiency = success rate / normalized cost
        return successRate / (1 + avgCost / 10);
    }

    /**
     * Calculate market opportunity coverage
     */
    private calculateMarketOpportunityCoverage(): number {
        const marketAtoms = Array.from(this.attentionSpace.values()).filter(
            atom =>
                atom.type === 'market_opportunity' || (atom.type === 'ingredient' && atom.content.market_opportunity),
        );

        return Math.min(1.0, marketAtoms.length / this.marketOpportunities.size);
    }

    /**
     * Calculate regulatory compliance attention level
     */
    private calculateRegulatoryComplianceLevel(): number {
        const regulatoryAtoms = Array.from(this.attentionSpace.values()).filter(atom => atom.type === 'constraint');

        const avgAttention =
            regulatoryAtoms.length > 0
                ? regulatoryAtoms.reduce((sum, atom) => sum + atom.attention_value, 0) / regulatoryAtoms.length
                : 0;

        return Math.min(1.0, avgAttention / 500); // Normalize to 0-1
    }
}
