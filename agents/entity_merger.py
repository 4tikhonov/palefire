#!/usr/bin/env python3
"""
Entity Merger

Merges spaCy-extracted entities with LLM-verified entities using robust matching.
Handles text variations, normalization, and proper status assignment.
"""

import re
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class EntityMerger:
    """
    Merges spaCy entities with LLM verification results.
    """

    def __init__(self):
        """Initialize the entity merger."""
        pass

    def merge_entities(self, spaCy_entities: List[Dict[str, Any]],
                      llm_entities: List[Dict[str, Any]],
                      llm_model: str = "unknown") -> List[Dict[str, Any]]:
        """
        Merge spaCy entities with LLM verification results.

        Args:
            spaCy_entities: List of entities from spaCy NER
            llm_entities: List of verified entities from LLM response parser
            llm_model: Name of the LLM model used (string for single model, will be overridden for multi-model)

        Returns:
            List of merged entity dictionaries with verification status
        """
        # Check if this is multi-model case (entities have source_model field)
        has_multiple_models = any(entity.get('source_model') for entity in llm_entities)
        model_names = [llm_model]  # Default single model

        if has_multiple_models:
            # Extract unique model names from entities
            model_names = list(set(entity.get('source_model', llm_model) for entity in llm_entities))
            return self._merge_multi_model_results(llm_entities, spaCy_entities, model_names)
        else:
            # Single model case
            return self._merge_single_model_results(spaCy_entities, llm_entities, llm_model)
        """
        Merge spaCy entities with LLM verification results.

        Args:
            spaCy_entities: List of entities from spaCy NER
            llm_entities: List of verified entities from LLM response parser
            llm_model: Name of the LLM model used for verification

        Returns:
            List of merged entity dictionaries with verification status
        """
        if not spaCy_entities:
            return []

        if not llm_entities:
            # No LLM verification, mark all as not verified
            return self._create_unverified_entities(spaCy_entities, llm_model)

        # Check if this is multi-model case (entities have source_model field)
        has_multiple_models = any(entity.get('source_model') for entity in llm_entities)
        model_names = [llm_model]  # Default single model

        if has_multiple_models:
            # Extract unique model names from entities
            model_names = list(set(entity.get('source_model', llm_model) for entity in llm_entities))
            return self._merge_multi_model_results(llm_entities, spaCy_entities, model_names)
        else:
            # Single model case
            return self._merge_single_model_results(spaCy_entities, llm_entities, llm_model)

    def _merge_multi_model_results(self, all_verified_entities: List[Dict[str, Any]],
                                  original_entities: List[Dict[str, Any]],
                                  model_names: List[str]) -> List[Dict[str, Any]]:
        """
        Merge verification results from multiple models using consensus voting.

        Args:
            all_verified_entities: All verified entities from all models
            original_entities: Original spaCy entities
            model_names: List of model names used

        Returns:
            Merged entities with consensus-based decisions
        """
        from collections import defaultdict

        # Group entities by text (case-insensitive, normalized)
        entity_groups = defaultdict(list)

        for entity in all_verified_entities:
            text = entity.get('text', '').strip()
            if not text:
                continue

            # Create normalized key for grouping
            normalized_key = text.lower()
            entity_groups[normalized_key].append(entity)

        merged_results = []
        total_models = len(model_names)

        for text_key, entities in entity_groups.items():
            if not entities:
                continue

            # Count votes for each decision type
            type_votes = defaultdict(int)
            verified_votes = 0
            remove_votes = 0
            uncertain_votes = 0
            reasoning_parts = []

            for entity in entities:
                entity_type = entity.get('type', 'UNKNOWN')
                is_verified = entity.get('verified', False)
                reasoning = entity.get('reasoning', '').strip()
                source_model = entity.get('source_model', 'unknown')

                if reasoning:
                    reasoning_parts.append(f"{source_model}: {reasoning}")

                if entity_type == 'REMOVE':
                    remove_votes += 1
                elif is_verified:
                    verified_votes += 1
                    type_votes[entity_type] += 1
                else:
                    uncertain_votes += 1

            # Determine consensus (majority vote)
            entity_text = entities[0]['text']
            majority_threshold = total_models // 2 + 1

            if remove_votes >= majority_threshold:
                # Majority consensus to remove
                merged_entity = {
                    'text': entity_text,
                    'type': 'REMOVE',
                    'reasoning': f"Majority consensus to remove ({remove_votes}/{total_models} models): {'; '.join(reasoning_parts)}",
                    'verified': True,
                    'new_entity': False,
                    'source_models': [e.get('source_model') for e in entities],
                    'consensus': f"Removed by {remove_votes}/{total_models} models"
                }
            elif verified_votes >= majority_threshold:
                # Majority consensus to verify with most common type
                most_common_type = max(type_votes.items(), key=lambda x: x[1])[0] if type_votes else 'UNKNOWN'

                merged_entity = {
                    'text': entity_text,
                    'type': most_common_type,
                    'reasoning': f"Majority consensus to verify ({verified_votes}/{total_models} models): {'; '.join(reasoning_parts)}",
                    'verified': True,
                    'new_entity': entities[0].get('new_entity', True),
                    'source_models': [e.get('source_model') for e in entities],
                    'consensus': f"Verified as {most_common_type} by {verified_votes}/{total_models} models"
                }

                # Preserve position info if available
                for entity in entities:
                    if 'start' in entity and 'end' in entity:
                        merged_entity.update({
                            'start': entity['start'],
                            'end': entity['end']
                        })
                        break
            else:
                # No clear consensus
                merged_entity = {
                    'text': entity_text,
                    'type': entities[0].get('type', 'UNKNOWN'),
                    'reasoning': f"No majority consensus ({verified_votes} verify, {remove_votes} remove, {uncertain_votes} uncertain out of {total_models} models): {'; '.join(reasoning_parts)}",
                    'verified': False,
                    'new_entity': entities[0].get('new_entity', True),
                    'source_models': [e.get('source_model') for e in entities],
                    'consensus': f"No consensus: {verified_votes}/{total_models} verify, {remove_votes}/{total_models} remove"
                }

            merged_results.append(merged_entity)

        logger.info(f"Multi-model consensus: {len(merged_results)} entities from {len(all_verified_entities)} inputs across {total_models} models")
        return merged_results

    def _merge_single_model_results(self, spaCy_entities: List[Dict[str, Any]],
                                  llm_entities: List[Dict[str, Any]],
                                  llm_model: str) -> List[Dict[str, Any]]:
        """
        Merge spaCy entities with LLM verification results from a single model.
        """
        logger.debug(f"Merging {len(spaCy_entities)} spaCy entities with {len(llm_entities)} LLM entities from {llm_model}")

        merged_entities = []

        for spaCy_entity in spaCy_entities:
            merged_entity = self._merge_single_entity(spaCy_entity, llm_entities, llm_model)
            merged_entities.append(merged_entity)

        logger.debug(f"Successfully merged {len(merged_entities)} entities from single model")
        return merged_entities

    def _merge_single_entity(self, spaCy_entity: Dict[str, Any],
                           llm_entities: List[Dict[str, Any]],
                           llm_model: str) -> Dict[str, Any]:
        """
        Merge a single spaCy entity with LLM verification results.
        """
        entity_text = spaCy_entity.get('text', '').strip()
        entity_type = spaCy_entity.get('type', 'UNKNOWN')

        # Find matching LLM entity
        matching_llm_entity = self._find_matching_llm_entity(entity_text, llm_entities)

        if matching_llm_entity:
            # Entity was verified by LLM
            return self._create_verified_entity(spaCy_entity, matching_llm_entity, llm_model)
        else:
            # Entity not verified by LLM
            return self._create_unverified_entity(spaCy_entity, llm_model)

    def _find_matching_llm_entity(self, entity_text: str,
                                llm_entities: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Find the best matching LLM entity for a spaCy entity text.

        Uses multiple matching strategies to handle text variations.
        """
        if not entity_text:
            return None

        for llm_entity in llm_entities:
            if not isinstance(llm_entity, dict):
                continue

            llm_text = llm_entity.get('text', '').strip()
            if not llm_text:
                continue

            # Try multiple matching strategies
            if self._texts_match(entity_text, llm_text):
                return llm_entity

        return None

    def _texts_match(self, text1: str, text2: str) -> bool:
        """
        Check if two texts match using multiple strategies.

        Handles case differences, spacing, and punctuation variations.
        """
        # Normalize both texts
        norm1 = self._normalize_text(text1)
        norm2 = self._normalize_text(text2)

        # Strategy 1: Exact match after normalization
        if norm1 == norm2:
            return True

        # Strategy 2: Exact match ignoring case and stripping
        if text1.strip().lower() == text2.strip().lower():
            return True

        # Strategy 3: Match without spaces (handles spacing variations)
        if norm1.replace(' ', '') == norm2.replace(' ', ''):
            return True

        # Strategy 4: Match without punctuation
        no_punct1 = re.sub(r'[^\w\s]', '', text1.strip().lower())
        no_punct2 = re.sub(r'[^\w\s]', '', text2.strip().lower())
        if no_punct1 == no_punct2:
            return True

        # Strategy 5: Match without spaces and punctuation
        if no_punct1.replace(' ', '') == no_punct2.replace(' ', ''):
            return True

        return False

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for comparison.
        """
        if not text:
            return ""

        # Convert to lowercase, normalize whitespace
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)  # Normalize spaces
        return text.strip()

    def _create_verified_entity(self, spaCy_entity: Dict[str, Any],
                              llm_entity: Dict[str, Any],
                              llm_model: str) -> Dict[str, Any]:
        """
        Create a verified entity from spaCy and LLM data.
        """
        # Use LLM's type and reasoning, but preserve spaCy's position info
        entity_type = llm_entity.get('type', spaCy_entity.get('type', 'UNKNOWN'))
        reasoning = llm_entity.get('reasoning', '').strip()

        # Only consider verified if LLM provided specific reasoning
        if reasoning and reasoning != 'No reasoning provided by LLM':
            status = 'verified'
        else:
            status = 'Not verified'
            reasoning = f"Not verified by LLM model {llm_model} - using spaCy classification ({entity_type})"

        return {
            'text': spaCy_entity.get('text', ''),
            'type': entity_type,
            'start': spaCy_entity.get('start'),
            'end': spaCy_entity.get('end'),
            'label': spaCy_entity.get('label', entity_type),
            'reasoning': reasoning,
            'status': status,
            'verified': True,
            'source': 'llm_verified'
        }

    def _create_unverified_entity(self, spaCy_entity: Dict[str, Any],
                                llm_model: str) -> Dict[str, Any]:
        """
        Create an unverified entity from spaCy data only.
        """
        entity_type = spaCy_entity.get('type', 'UNKNOWN')

        return {
            'text': spaCy_entity.get('text', ''),
            'type': entity_type,
            'start': spaCy_entity.get('start'),
            'end': spaCy_entity.get('end'),
            'label': spaCy_entity.get('label', entity_type),
            'reasoning': f"Not verified by LLM model {llm_model} - using spaCy classification ({entity_type})",
            'status': 'Not verified',
            'verified': False,
            'source': 'spacy_only'
        }

    def _create_unverified_entities(self, spaCy_entities: List[Dict[str, Any]],
                                  llm_model: str) -> List[Dict[str, Any]]:
        """
        Create unverified entities for all spaCy entities when no LLM verification available.
        """
        return [self._create_unverified_entity(entity, llm_model) for entity in spaCy_entities]

    def get_merge_statistics(self, spaCy_entities: List[Dict[str, Any]],
                           llm_entities: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Get statistics about the merge operation.
        """
        if not spaCy_entities:
            return {'total_spacy': 0, 'total_llm': len(llm_entities or []), 'matched': 0, 'unmatched': 0}

        matched = 0
        for spaCy_entity in spaCy_entities:
            entity_text = spaCy_entity.get('text', '').strip()
            if self._find_matching_llm_entity(entity_text, llm_entities or []):
                matched += 1

        return {
            'total_spacy': len(spaCy_entities),
            'total_llm': len(llm_entities or []),
            'matched': matched,
            'unmatched': len(spaCy_entities) - matched
        }
