"""Tests für Arrangement-Schemas"""

import pytest
from pydantic import ValidationError
from typing import List, Dict, Any

from schemas.arrangement import (
    ArrangementBase, ArrangementCreate, ArrangementUpdate, ArrangementResponse,
    ArrangementSection, ArrangementStem, ArrangementTransition,
    ArrangementStructure, ArrangementMetadata
)


class TestArrangementBase:
    """Tests für ArrangementBase-Schema"""

    @pytest.mark.unit
    def test_arrangement_base_creation(self):
        """Test: ArrangementBase erstellen"""
        arrangement_data = {
            "bpm": 128,
            "total_bars": 128,
            "track_structure": {"intro": 16, "drop": 64, "outro": 16},
            "stems": [1, 2, 3]
        }
        
        arrangement = ArrangementBase(**arrangement_data)
        
        assert arrangement.bpm == 128
        assert arrangement.total_bars == 128
        assert arrangement.track_structure["drop"] == 64
        assert arrangement.stems == [1, 2, 3]

    @pytest.mark.unit
    def test_arrangement_base_validation(self):
        """Test: ArrangementBase-Validierung"""
        with pytest.raises(ValidationError):
            ArrangementBase(bpm=120, total_bars=128, track_structure={}, stems=[])

        with pytest.raises(ValidationError):
            ArrangementBase(bpm=120, total_bars=128, track_structure={"intro": 16}, stems=None)


class TestArrangementCreate:
    """Tests für ArrangementCreate-Schema"""

    @pytest.mark.unit
    def test_arrangement_create_basic(self):
        """Test: Grundlegende ArrangementCreate-Erstellung"""
        create_data = {
            "bpm": 130,
            "total_bars": 128,
            "track_structure": {"main": 128},
            "stems": [10, 11, 12]
        }
        
        arrangement_create = ArrangementCreate(**create_data)
        
        assert arrangement_create.bpm == 130
        assert arrangement_create.total_bars == 128
        assert arrangement_create.stems == [10, 11, 12]


class TestArrangementUpdate:
    """Tests für ArrangementUpdate-Schema"""

    @pytest.mark.unit
    def test_arrangement_update_partial(self):
        """Test: Partielle ArrangementUpdate"""
        update_data = {
            "bpm": 125,
            "stems": [1, 2, 3, 4]
        }
        
        arrangement_update = ArrangementUpdate(**update_data)
        
        assert arrangement_update.bpm == 125
        assert arrangement_update.stems == [1, 2, 3, 4]
        assert arrangement_update.total_bars is None


class TestArrangementSection:
    """Tests für ArrangementSection-Schema"""

    @pytest.mark.unit
    def test_arrangement_section_basic(self):
        """Test: Grundlegende ArrangementSection"""
        section_data = {
            "section": "drop",
            "bars": 64,
            "stem_queries": [{"category": "kick", "mood": "heavy"}]
        }
        
        section = ArrangementSection(**section_data)
        
        assert section.section == "drop"
        assert section.bars == 64
        assert len(section.stem_queries) == 1
        assert section.stem_queries[0]["mood"] == "heavy"


class TestArrangementStem:
    """Tests für ArrangementStem-Schema"""

    @pytest.mark.unit
    def test_arrangement_stem_basic(self):
        """Test: Grundlegende ArrangementStem"""
        stem_data = {
            "stem_id": 123,
            "start_offset_bars": 16,
            "duration_bars": 32
        }
        
        arrangement_stem = ArrangementStem(**stem_data)
        
        assert arrangement_stem.stem_id == 123
        assert arrangement_stem.start_offset_bars == 16
        assert arrangement_stem.duration_bars == 32


class TestArrangementTransition:
    """Tests für ArrangementTransition-Schema"""

    @pytest.mark.unit
    def test_arrangement_transition_basic(self):
        """Test: Grundlegende ArrangementTransition"""
        transition_data = {
            "from_section": "buildup",
            "to_section": "drop",
            "type": "riser",
            "duration_bars": 4
        }
        
        transition = ArrangementTransition(**transition_data)
        
        assert transition.from_section == "buildup"
        assert transition.to_section == "drop"
        assert transition.type == "riser"
        assert transition.duration_bars == 4


class TestArrangementStructure:
    """Tests für ArrangementStructure-Schema"""

    @pytest.mark.unit
    def test_arrangement_structure_basic(self):
        """Test: Grundlegende ArrangementStructure"""
        structure_data = {
            "sections": [
                {"section": "intro", "bars": 16, "stem_queries": []},
                {"section": "drop", "bars": 64, "stem_queries": []}
            ],
            "transitions": [
                {"from_section": "intro", "to_section": "drop", "type": "impact", "duration_bars": 1}
            ]
        }
        
        structure = ArrangementStructure(**structure_data)
        
        assert len(structure.sections) == 2
        assert len(structure.transitions) == 1
        assert structure.sections[1].section == "drop"
