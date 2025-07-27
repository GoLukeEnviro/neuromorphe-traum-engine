"""Tests für Arrangement-Schemas"""

import pytest
from datetime import datetime, timedelta
from typing import List, Dict, Any
from unittest.mock import MagicMock

from src.schemas.arrangement import (
    ArrangementBase, ArrangementCreate, ArrangementUpdate, ArrangementResponse,
    ArrangementSection, ArrangementStem, ArrangementTransition,
    ArrangementStructure, ArrangementMetadata, ArrangementAnalysis,
    ArrangementExport, ArrangementTemplate, ArrangementBatch
)
from src.core.exceptions import ValidationError


class TestArrangementBase:
    """Tests für ArrangementBase-Schema"""
    
    @pytest.mark.unit
    def test_arrangement_base_creation(self):
        """Test: ArrangementBase erstellen"""
        arrangement_data = {
            "title": "Test Arrangement",
            "description": "A test arrangement for unit testing",
            "genre": "techno",
            "bpm": 128.0,
            "key": "Am",
            "duration": 300.0,
            "structure_type": "intro-buildup-drop-breakdown-outro",
            "tags": ["dark", "driving", "peak-time"]
        }
        
        arrangement = ArrangementBase(**arrangement_data)
        
        assert arrangement.title == "Test Arrangement"
        assert arrangement.genre == "techno"
        assert arrangement.bpm == 128.0
        assert arrangement.duration == 300.0
        assert arrangement.tags == ["dark", "driving", "peak-time"]
    
    @pytest.mark.unit
    def test_arrangement_base_validation(self):
        """Test: ArrangementBase-Validierung"""
        # Gültige Daten
        valid_data = {
            "title": "Valid Arrangement",
            "bpm": 120.0,
            "duration": 180.0
        }
        
        arrangement = ArrangementBase(**valid_data)
        assert arrangement.title == "Valid Arrangement"
        
        # Ungültige BPM
        with pytest.raises(ValidationError):
            ArrangementBase(title="Test", bpm=0)
        
        with pytest.raises(ValidationError):
            ArrangementBase(title="Test", bpm=300)  # Zu hoch
        
        # Ungültige Dauer
        with pytest.raises(ValidationError):
            ArrangementBase(title="Test", duration=-10.0)
        
        # Zu kurze Dauer
        with pytest.raises(ValidationError):
            ArrangementBase(title="Test", duration=5.0)  # Min 10 Sekunden
    
    @pytest.mark.unit
    def test_arrangement_base_optional_fields(self):
        """Test: Optionale Felder in ArrangementBase"""
        minimal_data = {
            "title": "Minimal Arrangement"
        }
        
        arrangement = ArrangementBase(**minimal_data)
        
        assert arrangement.title == "Minimal Arrangement"
        assert arrangement.description is None
        assert arrangement.genre is None
        assert arrangement.bpm is None
        assert arrangement.key is None
        assert arrangement.tags == []
        assert arrangement.metadata == {}


class TestArrangementCreate:
    """Tests für ArrangementCreate-Schema"""
    
    @pytest.mark.unit
    def test_arrangement_create_basic(self):
        """Test: Grundlegende ArrangementCreate-Erstellung"""
        create_data = {
            "title": "New Arrangement",
            "prompt": "Create a dark techno track with heavy kicks and atmospheric pads",
            "genre": "techno",
            "target_duration": 360.0,
            "target_bpm": 130.0
        }
        
        arrangement_create = ArrangementCreate(**create_data)
        
        assert arrangement_create.title == "New Arrangement"
        assert arrangement_create.prompt == "Create a dark techno track with heavy kicks and atmospheric pads"
        assert arrangement_create.target_duration == 360.0
        assert arrangement_create.target_bpm == 130.0
    
    @pytest.mark.unit
    def test_arrangement_create_with_stems(self):
        """Test: ArrangementCreate mit spezifischen Stems"""
        create_data = {
            "title": "Stem-based Arrangement",
            "stem_ids": ["stem_123", "stem_456", "stem_789"],
            "stem_preferences": {
                "kick_weight": 0.8,
                "bass_weight": 0.7,
                "lead_weight": 0.6
            }
        }
        
        arrangement_create = ArrangementCreate(**create_data)
        
        assert arrangement_create.stem_ids == ["stem_123", "stem_456", "stem_789"]
        assert arrangement_create.stem_preferences["kick_weight"] == 0.8
    
    @pytest.mark.unit
    def test_arrangement_create_structure_options(self):
        """Test: ArrangementCreate mit Struktur-Optionen"""
        create_data = {
            "title": "Structured Arrangement",
            "structure_template": "classic_techno",
            "auto_generate": True,
            "use_ai_analysis": True,
            "creativity_level": 0.7,
            "variation_intensity": 0.5
        }
        
        arrangement_create = ArrangementCreate(**create_data)
        
        assert arrangement_create.structure_template == "classic_techno"
        assert arrangement_create.auto_generate == True
        assert arrangement_create.creativity_level == 0.7
    
    @pytest.mark.unit
    def test_arrangement_create_validation(self):
        """Test: ArrangementCreate-Validierung"""
        # Title ist erforderlich
        with pytest.raises(ValidationError):
            ArrangementCreate()
        
        # Ungültige Creativity-Level
        with pytest.raises(ValidationError):
            ArrangementCreate(title="Test", creativity_level=1.5)
        
        # Ungültige Variation-Intensity
        with pytest.raises(ValidationError):
            ArrangementCreate(title="Test", variation_intensity=-0.1)


class TestArrangementUpdate:
    """Tests für ArrangementUpdate-Schema"""
    
    @pytest.mark.unit
    def test_arrangement_update_partial(self):
        """Test: Partielle ArrangementUpdate"""
        update_data = {
            "title": "Updated Title",
            "bpm": 125.0,
            "tags": ["updated", "modified"]
        }
        
        arrangement_update = ArrangementUpdate(**update_data)
        
        assert arrangement_update.title == "Updated Title"
        assert arrangement_update.bpm == 125.0
        assert arrangement_update.tags == ["updated", "modified"]
        assert arrangement_update.genre is None  # Nicht gesetzt
    
    @pytest.mark.unit
    def test_arrangement_update_structure(self):
        """Test: Struktur-Update in ArrangementUpdate"""
        update_data = {
            "structure_type": "custom",
            "sections": [
                {
                    "name": "intro",
                    "start_time": 0.0,
                    "duration": 32.0,
                    "type": "intro"
                },
                {
                    "name": "main",
                    "start_time": 32.0,
                    "duration": 128.0,
                    "type": "main"
                }
            ]
        }
        
        arrangement_update = ArrangementUpdate(**update_data)
        
        assert arrangement_update.structure_type == "custom"
        assert len(arrangement_update.sections) == 2
        assert arrangement_update.sections[0]["name"] == "intro"


class TestArrangementResponse:
    """Tests für ArrangementResponse-Schema"""
    
    @pytest.mark.unit
    def test_arrangement_response_complete(self):
        """Test: Vollständige ArrangementResponse"""
        response_data = {
            "id": "arrangement_123",
            "title": "Response Arrangement",
            "genre": "house",
            "bpm": 124.0,
            "duration": 240.0,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "status": "completed",
            "progress": 100.0,
            "stem_count": 8,
            "section_count": 5
        }
        
        arrangement_response = ArrangementResponse(**response_data)
        
        assert arrangement_response.id == "arrangement_123"
        assert arrangement_response.title == "Response Arrangement"
        assert arrangement_response.status == "completed"
        assert arrangement_response.stem_count == 8
        assert isinstance(arrangement_response.created_at, datetime)
    
    @pytest.mark.unit
    def test_arrangement_response_with_sections(self):
        """Test: ArrangementResponse mit Sections"""
        response_data = {
            "id": "arrangement_456",
            "title": "Sectioned Arrangement",
            "sections": [
                {
                    "id": "section_1",
                    "name": "intro",
                    "start_time": 0.0,
                    "duration": 32.0,
                    "type": "intro",
                    "stems": []
                },
                {
                    "id": "section_2",
                    "name": "buildup",
                    "start_time": 32.0,
                    "duration": 64.0,
                    "type": "buildup",
                    "stems": []
                }
            ],
            "transitions": [
                {
                    "from_section": "section_1",
                    "to_section": "section_2",
                    "type": "crossfade",
                    "duration": 4.0
                }
            ]
        }
        
        arrangement_response = ArrangementResponse(**response_data)
        
        assert len(arrangement_response.sections) == 2
        assert len(arrangement_response.transitions) == 1
        assert arrangement_response.sections[0]["name"] == "intro"
        assert arrangement_response.transitions[0]["type"] == "crossfade"


class TestArrangementSection:
    """Tests für ArrangementSection-Schema"""
    
    @pytest.mark.unit
    def test_arrangement_section_basic(self):
        """Test: Grundlegende ArrangementSection"""
        section_data = {
            "name": "drop",
            "start_time": 64.0,
            "duration": 128.0,
            "type": "drop",
            "energy_level": 9,
            "complexity": 8
        }
        
        section = ArrangementSection(**section_data)
        
        assert section.name == "drop"
        assert section.start_time == 64.0
        assert section.duration == 128.0
        assert section.type == "drop"
        assert section.energy_level == 9
        assert section.complexity == 8
    
    @pytest.mark.unit
    def test_arrangement_section_with_stems(self):
        """Test: ArrangementSection mit Stems"""
        section_data = {
            "name": "main",
            "start_time": 32.0,
            "duration": 96.0,
            "type": "main",
            "stems": [
                {
                    "stem_id": "kick_123",
                    "start_offset": 0.0,
                    "volume": 0.8,
                    "pan": 0.0,
                    "effects": {"reverb": 0.2}
                },
                {
                    "stem_id": "bass_456",
                    "start_offset": 8.0,
                    "volume": 0.7,
                    "pan": -0.1,
                    "effects": {"filter": {"type": "lowpass", "frequency": 800}}
                }
            ]
        }
        
        section = ArrangementSection(**section_data)
        
        assert len(section.stems) == 2
        assert section.stems[0]["stem_id"] == "kick_123"
        assert section.stems[1]["start_offset"] == 8.0
    
    @pytest.mark.unit
    def test_arrangement_section_validation(self):
        """Test: ArrangementSection-Validierung"""
        # Ungültige Energy-Level
        with pytest.raises(ValidationError):
            ArrangementSection(
                name="test",
                start_time=0.0,
                duration=32.0,
                energy_level=11  # Max ist 10
            )
        
        # Ungültige Dauer
        with pytest.raises(ValidationError):
            ArrangementSection(
                name="test",
                start_time=0.0,
                duration=-10.0  # Muss positiv sein
            )
        
        # Ungültige Start-Zeit
        with pytest.raises(ValidationError):
            ArrangementSection(
                name="test",
                start_time=-5.0,  # Muss >= 0 sein
                duration=32.0
            )


class TestArrangementStem:
    """Tests für ArrangementStem-Schema"""
    
    @pytest.mark.unit
    def test_arrangement_stem_basic(self):
        """Test: Grundlegende ArrangementStem"""
        stem_data = {
            "stem_id": "stem_123",
            "start_offset": 16.0,
            "volume": 0.75,
            "pan": 0.2,
            "mute": False,
            "solo": False
        }
        
        arrangement_stem = ArrangementStem(**stem_data)
        
        assert arrangement_stem.stem_id == "stem_123"
        assert arrangement_stem.start_offset == 16.0
        assert arrangement_stem.volume == 0.75
        assert arrangement_stem.pan == 0.2
        assert arrangement_stem.mute == False
        assert arrangement_stem.solo == False
    
    @pytest.mark.unit
    def test_arrangement_stem_with_effects(self):
        """Test: ArrangementStem mit Effekten"""
        stem_data = {
            "stem_id": "stem_456",
            "effects": {
                "reverb": {
                    "wet": 0.3,
                    "room_size": 0.7,
                    "damping": 0.5
                },
                "delay": {
                    "time": 0.25,
                    "feedback": 0.4,
                    "wet": 0.2
                },
                "eq": {
                    "low": 0.0,
                    "mid": 0.1,
                    "high": -0.1
                }
            }
        }
        
        arrangement_stem = ArrangementStem(**stem_data)
        
        assert "reverb" in arrangement_stem.effects
        assert "delay" in arrangement_stem.effects
        assert "eq" in arrangement_stem.effects
        assert arrangement_stem.effects["reverb"]["wet"] == 0.3
    
    @pytest.mark.unit
    def test_arrangement_stem_automation(self):
        """Test: ArrangementStem mit Automation"""
        stem_data = {
            "stem_id": "stem_789",
            "automation": {
                "volume": [
                    {"time": 0.0, "value": 0.0},
                    {"time": 4.0, "value": 0.8},
                    {"time": 28.0, "value": 0.8},
                    {"time": 32.0, "value": 0.0}
                ],
                "pan": [
                    {"time": 0.0, "value": -0.5},
                    {"time": 16.0, "value": 0.5}
                ]
            }
        }
        
        arrangement_stem = ArrangementStem(**stem_data)
        
        assert "volume" in arrangement_stem.automation
        assert "pan" in arrangement_stem.automation
        assert len(arrangement_stem.automation["volume"]) == 4
        assert arrangement_stem.automation["volume"][1]["value"] == 0.8
    
    @pytest.mark.unit
    def test_arrangement_stem_validation(self):
        """Test: ArrangementStem-Validierung"""
        # Ungültiges Volume
        with pytest.raises(ValidationError):
            ArrangementStem(stem_id="test", volume=1.5)  # Max ist 1.0
        
        with pytest.raises(ValidationError):
            ArrangementStem(stem_id="test", volume=-0.1)  # Min ist 0.0
        
        # Ungültiges Pan
        with pytest.raises(ValidationError):
            ArrangementStem(stem_id="test", pan=1.5)  # Max ist 1.0
        
        with pytest.raises(ValidationError):
            ArrangementStem(stem_id="test", pan=-1.5)  # Min ist -1.0
        
        # Ungültiger Start-Offset
        with pytest.raises(ValidationError):
            ArrangementStem(stem_id="test", start_offset=-1.0)  # Muss >= 0 sein


class TestArrangementTransition:
    """Tests für ArrangementTransition-Schema"""
    
    @pytest.mark.unit
    def test_arrangement_transition_basic(self):
        """Test: Grundlegende ArrangementTransition"""
        transition_data = {
            "from_section": "section_1",
            "to_section": "section_2",
            "type": "crossfade",
            "duration": 8.0,
            "curve": "linear"
        }
        
        transition = ArrangementTransition(**transition_data)
        
        assert transition.from_section == "section_1"
        assert transition.to_section == "section_2"
        assert transition.type == "crossfade"
        assert transition.duration == 8.0
        assert transition.curve == "linear"
    
    @pytest.mark.unit
    def test_arrangement_transition_with_effects(self):
        """Test: ArrangementTransition mit Effekten"""
        transition_data = {
            "from_section": "buildup",
            "to_section": "drop",
            "type": "filter_sweep",
            "duration": 4.0,
            "effects": {
                "filter": {
                    "type": "lowpass",
                    "start_frequency": 20000,
                    "end_frequency": 200,
                    "resonance": 0.7
                },
                "reverb": {
                    "start_wet": 0.1,
                    "end_wet": 0.8
                }
            }
        }
        
        transition = ArrangementTransition(**transition_data)
        
        assert transition.type == "filter_sweep"
        assert "filter" in transition.effects
        assert "reverb" in transition.effects
        assert transition.effects["filter"]["start_frequency"] == 20000
    
    @pytest.mark.unit
    def test_arrangement_transition_validation(self):
        """Test: ArrangementTransition-Validierung"""
        # Ungültige Dauer
        with pytest.raises(ValidationError):
            ArrangementTransition(
                from_section="a",
                to_section="b",
                duration=-1.0  # Muss positiv sein
            )
        
        # Gleiche Sections
        with pytest.raises(ValidationError):
            ArrangementTransition(
                from_section="same",
                to_section="same",  # Darf nicht gleich sein
                duration=2.0
            )
        
        # Ungültiger Transition-Type
        with pytest.raises(ValidationError):
            ArrangementTransition(
                from_section="a",
                to_section="b",
                type="invalid_type",
                duration=2.0
            )


class TestArrangementStructure:
    """Tests für ArrangementStructure-Schema"""
    
    @pytest.mark.unit
    def test_arrangement_structure_basic(self):
        """Test: Grundlegende ArrangementStructure"""
        structure_data = {
            "template_name": "classic_techno",
            "total_duration": 360.0,
            "section_definitions": [
                {"type": "intro", "min_duration": 16.0, "max_duration": 32.0},
                {"type": "buildup", "min_duration": 32.0, "max_duration": 64.0},
                {"type": "drop", "min_duration": 64.0, "max_duration": 128.0},
                {"type": "breakdown", "min_duration": 32.0, "max_duration": 64.0},
                {"type": "outro", "min_duration": 16.0, "max_duration": 32.0}
            ]
        }
        
        structure = ArrangementStructure(**structure_data)
        
        assert structure.template_name == "classic_techno"
        assert structure.total_duration == 360.0
        assert len(structure.section_definitions) == 5
        assert structure.section_definitions[0]["type"] == "intro"
    
    @pytest.mark.unit
    def test_arrangement_structure_with_rules(self):
        """Test: ArrangementStructure mit Regeln"""
        structure_data = {
            "template_name": "custom",
            "rules": {
                "energy_progression": "gradual_buildup",
                "complexity_variation": "moderate",
                "transition_density": "sparse",
                "stem_layering": "progressive"
            },
            "constraints": {
                "max_simultaneous_stems": 8,
                "min_section_duration": 16.0,
                "max_section_duration": 128.0,
                "required_sections": ["intro", "main", "outro"]
            }
        }
        
        structure = ArrangementStructure(**structure_data)
        
        assert structure.rules["energy_progression"] == "gradual_buildup"
        assert structure.constraints["max_simultaneous_stems"] == 8
        assert "intro" in structure.constraints["required_sections"]


class TestArrangementMetadata:
    """Tests für ArrangementMetadata-Schema"""
    
    @pytest.mark.unit
    def test_arrangement_metadata_creation(self):
        """Test: ArrangementMetadata erstellen"""
        metadata_data = {
            "created_by": "test_user",
            "creation_method": "ai_generated",
            "prompt_used": "Create a dark techno track",
            "ai_model_version": "v2.1",
            "generation_time": 45.2,
            "stem_sources": {
                "kick": "sample_pack_a",
                "bass": "synthesized",
                "lead": "sample_pack_b"
            },
            "processing_stats": {
                "stems_analyzed": 150,
                "stems_selected": 8,
                "similarity_threshold": 0.75
            }
        }
        
        metadata = ArrangementMetadata(**metadata_data)
        
        assert metadata.created_by == "test_user"
        assert metadata.creation_method == "ai_generated"
        assert metadata.generation_time == 45.2
        assert metadata.stem_sources["kick"] == "sample_pack_a"
        assert metadata.processing_stats["stems_analyzed"] == 150


class TestArrangementAnalysis:
    """Tests für ArrangementAnalysis-Schema"""
    
    @pytest.mark.unit
    def test_arrangement_analysis_complete(self):
        """Test: Vollständige ArrangementAnalysis"""
        analysis_data = {
            "arrangement_id": "arrangement_123",
            "analysis_type": "full",
            "status": "completed",
            "progress": 100.0,
            "started_at": datetime.now(),
            "completed_at": datetime.now(),
            "energy_curve": [2, 3, 5, 8, 9, 8, 6, 4, 2],
            "complexity_curve": [1, 2, 4, 7, 8, 7, 5, 3, 1],
            "harmonic_analysis": {
                "key_stability": 0.85,
                "chord_progressions": ["Am", "F", "C", "G"],
                "modulations": []
            },
            "rhythmic_analysis": {
                "tempo_stability": 0.95,
                "groove_consistency": 0.88,
                "syncopation_level": 0.3
            },
            "structural_coherence": 0.92,
            "mix_balance": {
                "frequency_distribution": {"low": 0.3, "mid": 0.4, "high": 0.3},
                "stereo_width": 0.7,
                "dynamic_range": 0.6
            }
        }
        
        analysis = ArrangementAnalysis(**analysis_data)
        
        assert analysis.arrangement_id == "arrangement_123"
        assert analysis.status == "completed"
        assert len(analysis.energy_curve) == 9
        assert analysis.harmonic_analysis["key_stability"] == 0.85
        assert analysis.structural_coherence == 0.92
    
    @pytest.mark.unit
    def test_arrangement_analysis_validation(self):
        """Test: ArrangementAnalysis-Validierung"""
        # Ungültiger Progress
        with pytest.raises(ValidationError):
            ArrangementAnalysis(
                arrangement_id="test",
                progress=150.0  # Max ist 100.0
            )
        
        # Ungültiger Status
        with pytest.raises(ValidationError):
            ArrangementAnalysis(
                arrangement_id="test",
                status="invalid_status"
            )


class TestArrangementExport:
    """Tests für ArrangementExport-Schema"""
    
    @pytest.mark.unit
    def test_arrangement_export_basic(self):
        """Test: Grundlegende ArrangementExport"""
        export_data = {
            "arrangement_id": "arrangement_123",
            "format": "wav",
            "quality": "high",
            "sample_rate": 44100,
            "bit_depth": 24,
            "normalize": True,
            "apply_limiter": True
        }
        
        export = ArrangementExport(**export_data)
        
        assert export.arrangement_id == "arrangement_123"
        assert export.format == "wav"
        assert export.quality == "high"
        assert export.sample_rate == 44100
        assert export.normalize == True
    
    @pytest.mark.unit
    def test_arrangement_export_with_options(self):
        """Test: ArrangementExport mit erweiterten Optionen"""
        export_data = {
            "arrangement_id": "arrangement_456",
            "format": "mp3",
            "quality": "medium",
            "export_stems": True,
            "stem_format": "wav",
            "include_metadata": True,
            "fade_in": 2.0,
            "fade_out": 4.0,
            "trim_silence": True,
            "custom_effects": {
                "master_eq": {"low": 0.1, "mid": 0.0, "high": 0.05},
                "master_compressor": {"ratio": 4.0, "threshold": -12.0}
            }
        }
        
        export = ArrangementExport(**export_data)
        
        assert export.export_stems == True
        assert export.fade_in == 2.0
        assert export.custom_effects["master_eq"]["low"] == 0.1
    
    @pytest.mark.unit
    def test_arrangement_export_validation(self):
        """Test: ArrangementExport-Validierung"""
        # Ungültiges Format
        with pytest.raises(ValidationError):
            ArrangementExport(
                arrangement_id="test",
                format="invalid_format"
            )
        
        # Ungültige Sample-Rate
        with pytest.raises(ValidationError):
            ArrangementExport(
                arrangement_id="test",
                sample_rate=1000  # Zu niedrig
            )
        
        # Ungültige Bit-Depth
        with pytest.raises(ValidationError):
            ArrangementExport(
                arrangement_id="test",
                bit_depth=7  # Nicht unterstützt
            )


class TestArrangementTemplate:
    """Tests für ArrangementTemplate-Schema"""
    
    @pytest.mark.unit
    def test_arrangement_template_creation(self):
        """Test: ArrangementTemplate erstellen"""
        template_data = {
            "name": "Dark Techno Template",
            "description": "A template for creating dark techno arrangements",
            "genre": "techno",
            "subgenre": "dark_techno",
            "typical_bpm_range": {"min": 125, "max": 135},
            "typical_duration": 360.0,
            "structure_pattern": "intro-buildup-drop-breakdown-drop-outro",
            "energy_progression": [2, 3, 5, 8, 9, 6, 9, 4, 2],
            "recommended_stems": {
                "kick": {"weight": 1.0, "tags": ["dark", "punchy"]},
                "bass": {"weight": 0.8, "tags": ["deep", "rolling"]},
                "lead": {"weight": 0.6, "tags": ["acid", "distorted"]}
            }
        }
        
        template = ArrangementTemplate(**template_data)
        
        assert template.name == "Dark Techno Template"
        assert template.genre == "techno"
        assert template.typical_bpm_range["min"] == 125
        assert len(template.energy_progression) == 9
        assert template.recommended_stems["kick"]["weight"] == 1.0


class TestArrangementBatch:
    """Tests für ArrangementBatch-Schema"""
    
    @pytest.mark.unit
    def test_arrangement_batch_creation(self):
        """Test: ArrangementBatch erstellen"""
        batch_data = {
            "arrangements": [
                {"title": "Arrangement 1", "prompt": "Dark techno track"},
                {"title": "Arrangement 2", "prompt": "Melodic house track"},
                {"title": "Arrangement 3", "prompt": "Minimal techno track"}
            ],
            "batch_id": "batch_123",
            "batch_settings": {
                "auto_generate": True,
                "use_ai_analysis": True,
                "creativity_level": 0.7
            }
        }
        
        batch = ArrangementBatch(**batch_data)
        
        assert len(batch.arrangements) == 3
        assert batch.batch_id == "batch_123"
        assert batch.batch_settings["creativity_level"] == 0.7
    
    @pytest.mark.unit
    def test_arrangement_batch_validation(self):
        """Test: ArrangementBatch-Validierung"""
        # Leere Arrangements-Liste
        with pytest.raises(ValidationError):
            ArrangementBatch(arrangements=[])
        
        # Zu viele Arrangements
        large_arrangements_list = [{"title": f"Arrangement {i}"} for i in range(51)]
        with pytest.raises(ValidationError):
            ArrangementBatch(arrangements=large_arrangements_list)  # Max 50


class TestArrangementSchemasIntegration:
    """Integrationstests für Arrangement-Schemas"""
    
    @pytest.mark.integration
    def test_arrangement_lifecycle_schemas(self):
        """Test: Vollständiger Arrangement-Lifecycle mit Schemas"""
        # 1. Arrangement erstellen
        create_data = {
            "title": "Lifecycle Test Arrangement",
            "prompt": "Create an energetic techno track with driving bassline",
            "genre": "techno",
            "target_duration": 300.0,
            "target_bpm": 128.0,
            "auto_generate": True
        }
        
        arrangement_create = ArrangementCreate(**create_data)
        assert arrangement_create.title == "Lifecycle Test Arrangement"
        
        # 2. Arrangement-Response simulieren
        response_data = {
            "id": "arrangement_lifecycle_123",
            "title": arrangement_create.title,
            "genre": arrangement_create.genre,
            "bpm": 128.0,
            "duration": 300.0,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "status": "completed",
            "progress": 100.0,
            "stem_count": 6,
            "section_count": 5
        }
        
        arrangement_response = ArrangementResponse(**response_data)
        assert arrangement_response.id == "arrangement_lifecycle_123"
        
        # 3. Arrangement aktualisieren
        update_data = {
            "description": "Updated description",
            "tags": ["energetic", "driving", "peak-time"]
        }
        
        arrangement_update = ArrangementUpdate(**update_data)
        assert arrangement_update.tags == ["energetic", "driving", "peak-time"]
        
        # 4. Analyse-Ergebnis
        analysis_data = {
            "arrangement_id": arrangement_response.id,
            "status": "completed",
            "progress": 100.0,
            "energy_curve": [2, 4, 6, 8, 9, 7, 5, 3, 2],
            "structural_coherence": 0.88
        }
        
        analysis = ArrangementAnalysis(**analysis_data)
        assert analysis.status == "completed"
        
        # 5. Export-Konfiguration
        export_data = {
            "arrangement_id": arrangement_response.id,
            "format": "wav",
            "quality": "high",
            "normalize": True
        }
        
        export = ArrangementExport(**export_data)
        assert export.format == "wav"
    
    @pytest.mark.performance
    def test_arrangement_schemas_performance(self):
        """Test: Performance der Arrangement-Schemas"""
        import time
        
        # Viele Arrangements erstellen
        start_time = time.time()
        
        arrangements = []
        for i in range(100):
            arrangement_data = {
                "title": f"Performance Test Arrangement {i}",
                "genre": "techno",
                "bpm": 120.0 + (i % 20),
                "duration": 240.0 + (i * 10),
                "tags": [f"tag_{i % 5}"],
                "metadata": {"test_id": i}
            }
            
            arrangement = ArrangementCreate(**arrangement_data)
            arrangements.append(arrangement)
        
        creation_time = time.time() - start_time
        
        assert len(arrangements) == 100
        assert creation_time < 2.0  # Sollte unter 2 Sekunden dauern
        
        # Komplexe Arrangements mit Sections
        start_time = time.time()
        
        complex_arrangements = []
        for i in range(10):
            sections = []
            for j in range(5):  # 5 Sections pro Arrangement
                section = {
                    "name": f"section_{j}",
                    "start_time": j * 32.0,
                    "duration": 32.0,
                    "type": ["intro", "buildup", "drop", "breakdown", "outro"][j],
                    "stems": [
                        {
                            "stem_id": f"stem_{k}",
                            "volume": 0.8,
                            "pan": 0.0
                        } for k in range(3)  # 3 Stems pro Section
                    ]
                }
                sections.append(section)
            
            arrangement_data = {
                "title": f"Complex Arrangement {i}",
                "sections": sections
            }
            
            arrangement = ArrangementResponse(
                id=f"complex_{i}",
                **arrangement_data
            )
            complex_arrangements.append(arrangement)
        
        complex_creation_time = time.time() - start_time
        
        assert len(complex_arrangements) == 10
        assert complex_creation_time < 1.0  # Sollte unter 1 Sekunde dauern