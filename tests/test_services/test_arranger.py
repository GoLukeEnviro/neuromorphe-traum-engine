"""Tests für ArrangerService"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import numpy as np
from pathlib import Path

from src.services.arranger import ArrangerService
from src.core.config import Settings


class TestArrangerService:
    """Test-Suite für ArrangerService"""
    
    @pytest.fixture
    def arranger(self) -> ArrangerService:
        """ArrangerService-Instanz für Tests"""
        with patch('src.services.arranger.DatabaseService') as mock_db_service:
            arranger = ArrangerService()
            arranger.db_service = mock_db_service.return_value
            return arranger
    
    @pytest.mark.unit
    def test_initialization(self):
        """Test: ArrangerService-Initialisierung"""
        with patch('src.services.arranger.DatabaseService') as mock_db_service:
            arranger = ArrangerService()
            assert arranger.parser is not None
            assert arranger.db_service is not None
            mock_db_service.assert_called_once()
    
    @pytest.mark.unit
    async def test_create_arrangement_from_prompt(self, arranger: ArrangerService, test_db_session, sample_text_prompts):
        """Test: Arrangement aus Text-Prompt erstellen"""
        prompt = sample_text_prompts[0]  # "Dark atmospheric techno with heavy bass"
        
        with patch.object(arranger, '_analyze_prompt') as mock_analyze, \
             patch.object(arranger, '_select_stems') as mock_select, \
             patch.object(arranger, '_create_structure') as mock_structure:
            
            # Mock Prompt-Analyse
            mock_analyze.return_value = {
                "genre": "techno",
                "mood": "dark",
                "energy": 0.8,
                "tempo": 128,
                "elements": ["bass", "drums", "synth"]
            }
            
            # Mock Stem-Auswahl
            mock_select.return_value = [
                {"id": 1, "type": "kick", "similarity": 0.9},
                {"id": 2, "type": "bass", "similarity": 0.85},
                {"id": 3, "type": "synth", "similarity": 0.8}
            ]
            
            # Mock Struktur-Erstellung
            mock_structure.return_value = {
                "sections": [
                    {"name": "intro", "start": 0, "duration": 32, "stems": [1]},
                    {"name": "buildup", "start": 32, "duration": 32, "stems": [1, 2]},
                    {"name": "drop", "start": 64, "duration": 64, "stems": [1, 2, 3]}
                ],
                "total_duration": 160
            }
            
            result = await arranger.create_arrangement(
                prompt=prompt,
                duration=180,
                session=test_db_session
            )
            
            assert "arrangement_id" in result
            assert "structure" in result
            assert "stems" in result
            assert "metadata" in result
            
            mock_analyze.assert_called_once_with(prompt)
            mock_select.assert_called_once()
            mock_structure.assert_called_once()
    
    @pytest.mark.unit
    async def test_analyze_prompt_techno(self, arranger: ArrangerService):
        """Test: Techno-Prompt-Analyse"""
        prompt = "Dark atmospheric techno with heavy bass and industrial elements"
        
        with patch.object(arranger.neuro_analyzer, 'analyze_text_prompt') as mock_analyze:
            mock_analyze.return_value = {
                "embeddings": np.random.rand(512).tolist(),
                "prompt": prompt
            }
            
            analysis = await arranger._analyze_prompt(prompt)
            
            assert analysis["genre"] == "techno"
            assert "dark" in analysis["mood"] or "atmospheric" in analysis["mood"]
            assert "bass" in analysis["elements"]
            assert "industrial" in analysis["elements"]
            assert 120 <= analysis["tempo"] <= 140  # Typischer Techno-Bereich
    
    @pytest.mark.unit
    async def test_analyze_prompt_house(self, arranger: ArrangerService):
        """Test: House-Prompt-Analyse"""
        prompt = "Uplifting house music with piano melodies and vocal chops"
        
        with patch.object(arranger.neuro_analyzer, 'analyze_text_prompt') as mock_analyze:
            mock_analyze.return_value = {
                "embeddings": np.random.rand(512).tolist(),
                "prompt": prompt
            }
            
            analysis = await arranger._analyze_prompt(prompt)
            
            assert analysis["genre"] == "house"
            assert "uplifting" in analysis["mood"]
            assert "piano" in analysis["elements"]
            assert "vocal" in analysis["elements"]
            assert 120 <= analysis["tempo"] <= 130  # Typischer House-Bereich
    
    @pytest.mark.unit
    async def test_select_stems_by_similarity(self, arranger: ArrangerService, test_db_session):
        """Test: Stem-Auswahl nach Ähnlichkeit"""
        prompt_analysis = {
            "genre": "techno",
            "mood": "dark",
            "elements": ["kick", "bass", "synth"],
            "embeddings": np.random.rand(512).tolist()
        }
        
        with patch.object(arranger.neuro_analyzer, 'get_similar_stems') as mock_similar:
            mock_similar.return_value = [
                {"id": 1, "type": "kick", "similarity": 0.95, "genre": "techno"},
                {"id": 2, "type": "bass", "similarity": 0.90, "genre": "techno"},
                {"id": 3, "type": "synth", "similarity": 0.85, "genre": "techno"},
                {"id": 4, "type": "hihat", "similarity": 0.80, "genre": "techno"}
            ]
            
            selected_stems = await arranger._select_stems(
                prompt_analysis,
                session=test_db_session,
                max_stems=8
            )
            
            assert len(selected_stems) <= 8
            assert all(stem["similarity"] >= 0.7 for stem in selected_stems)  # Mindest-Ähnlichkeit
            
            # Sollte nach Ähnlichkeit sortiert sein
            similarities = [stem["similarity"] for stem in selected_stems]
            assert similarities == sorted(similarities, reverse=True)
    
    @pytest.mark.unit
    async def test_create_techno_structure(self, arranger: ArrangerService):
        """Test: Techno-Struktur-Erstellung"""
        prompt_analysis = {
            "genre": "techno",
            "mood": "dark",
            "energy": 0.8,
            "tempo": 128
        }
        
        selected_stems = [
            {"id": 1, "type": "kick", "energy": 0.9},
            {"id": 2, "type": "bass", "energy": 0.8},
            {"id": 3, "type": "synth", "energy": 0.7},
            {"id": 4, "type": "hihat", "energy": 0.6}
        ]
        
        structure = await arranger._create_structure(
            prompt_analysis,
            selected_stems,
            duration=180
        )
        
        assert "sections" in structure
        assert "total_duration" in structure
        assert structure["total_duration"] <= 180
        
        sections = structure["sections"]
        assert len(sections) >= 3  # Mindestens Intro, Main, Outro
        
        # Erste Sektion sollte Intro sein
        assert sections[0]["name"] in ["intro", "buildup"]
        assert sections[0]["start"] == 0
        
        # Sektionen sollten lückenlos aneinander anschließen
        for i in range(1, len(sections)):
            expected_start = sections[i-1]["start"] + sections[i-1]["duration"]
            assert sections[i]["start"] == expected_start
    
    @pytest.mark.unit
    async def test_create_house_structure(self, arranger: ArrangerService):
        """Test: House-Struktur-Erstellung"""
        prompt_analysis = {
            "genre": "house",
            "mood": "uplifting",
            "energy": 0.7,
            "tempo": 125
        }
        
        selected_stems = [
            {"id": 1, "type": "kick", "energy": 0.8},
            {"id": 2, "type": "piano", "energy": 0.6},
            {"id": 3, "type": "vocal", "energy": 0.7}
        ]
        
        structure = await arranger._create_structure(
            prompt_analysis,
            selected_stems,
            duration=240
        )
        
        sections = structure["sections"]
        
        # House sollte typische Struktur haben
        section_names = [s["name"] for s in sections]
        assert "verse" in section_names or "breakdown" in section_names
        assert "chorus" in section_names or "drop" in section_names
    
    @pytest.mark.unit
    async def test_apply_transitions(self, arranger: ArrangerService):
        """Test: Übergänge zwischen Sektionen"""
        structure = {
            "sections": [
                {"name": "intro", "start": 0, "duration": 32, "stems": [1]},
                {"name": "buildup", "start": 32, "duration": 32, "stems": [1, 2]},
                {"name": "drop", "start": 64, "duration": 64, "stems": [1, 2, 3]}
            ]
        }
        
        enhanced_structure = await arranger._apply_transitions(structure)
        
        # Übergänge sollten hinzugefügt worden sein
        for section in enhanced_structure["sections"]:
            if "transitions" in section:
                assert "fade_in" in section["transitions"] or "fade_out" in section["transitions"]
    
    @pytest.mark.unit
    async def test_validate_arrangement(self, arranger: ArrangerService):
        """Test: Arrangement-Validierung"""
        # Gültiges Arrangement
        valid_arrangement = {
            "arrangement_id": "test_123",
            "structure": {
                "sections": [
                    {"name": "intro", "start": 0, "duration": 32, "stems": [1]},
                    {"name": "main", "start": 32, "duration": 96, "stems": [1, 2, 3]}
                ],
                "total_duration": 128
            },
            "stems": [1, 2, 3],
            "metadata": {"genre": "techno", "tempo": 128}
        }
        
        # Ungültiges Arrangement (überlappende Sektionen)
        invalid_arrangement = {
            "arrangement_id": "test_456",
            "structure": {
                "sections": [
                    {"name": "intro", "start": 0, "duration": 40, "stems": [1]},
                    {"name": "main", "start": 30, "duration": 60, "stems": [1, 2]}  # Überlappung!
                ],
                "total_duration": 90
            },
            "stems": [1, 2],
            "metadata": {"genre": "techno"}
        }
        
        assert await arranger._validate_arrangement(valid_arrangement) == True
        assert await arranger._validate_arrangement(invalid_arrangement) == False
    
    @pytest.mark.unit
    async def test_optimize_stem_placement(self, arranger: ArrangerService):
        """Test: Optimierung der Stem-Platzierung"""
        stems = [
            {"id": 1, "type": "kick", "energy": 0.9, "complexity": 0.3},
            {"id": 2, "type": "bass", "energy": 0.8, "complexity": 0.5},
            {"id": 3, "type": "synth", "energy": 0.7, "complexity": 0.8},
            {"id": 4, "type": "fx", "energy": 0.4, "complexity": 0.9}
        ]
        
        sections = [
            {"name": "intro", "target_energy": 0.3, "max_stems": 2},
            {"name": "buildup", "target_energy": 0.6, "max_stems": 3},
            {"name": "drop", "target_energy": 0.9, "max_stems": 4}
        ]
        
        optimized_sections = await arranger._optimize_stem_placement(stems, sections)
        
        # Intro sollte weniger energetische Stems haben
        intro_stems = optimized_sections[0]["stems"]
        assert len(intro_stems) <= 2
        
        # Drop sollte die energetischsten Stems haben
        drop_stems = optimized_sections[2]["stems"]
        assert 1 in drop_stems  # Kick sollte im Drop sein
        assert 2 in drop_stems  # Bass sollte im Drop sein
    
    @pytest.mark.unit
    async def test_error_handling_invalid_prompt(self, arranger: ArrangerService, test_db_session):
        """Test: Fehlerbehandlung bei ungültigem Prompt"""
        with pytest.raises(ValueError):
            await arranger.create_arrangement(
                prompt="",  # Leerer Prompt
                duration=180,
                session=test_db_session
            )
    
    @pytest.mark.unit
    async def test_error_handling_invalid_duration(self, arranger: ArrangerService, test_db_session):
        """Test: Fehlerbehandlung bei ungültiger Dauer"""
        with pytest.raises(ValueError):
            await arranger.create_arrangement(
                prompt="Test prompt",
                duration=0,  # Ungültige Dauer
                session=test_db_session
            )
    
    @pytest.mark.unit
    async def test_genre_detection_accuracy(self, arranger: ArrangerService):
        """Test: Genauigkeit der Genre-Erkennung"""
        test_cases = [
            ("Dark techno with industrial sounds", "techno"),
            ("Uplifting house with piano", "house"),
            ("Minimal techno with hypnotic patterns", "techno"),
            ("Deep house with vocal elements", "house"),
            ("Ambient soundscape with drones", "ambient")
        ]
        
        for prompt, expected_genre in test_cases:
            with patch.object(arranger.neuro_analyzer, 'analyze_text_prompt'):
                analysis = await arranger._analyze_prompt(prompt)
                assert analysis["genre"] == expected_genre
    
    @pytest.mark.performance
    async def test_performance_large_stem_database(self, arranger: ArrangerService, test_db_session):
        """Test: Performance bei großer Stem-Datenbank"""
        import time
        
        # Große Anzahl von Stems simulieren
        with patch.object(arranger.neuro_analyzer, 'get_similar_stems') as mock_similar:
            # 1000 Stems simulieren
            large_stem_list = [
                {"id": i, "type": "kick", "similarity": 0.9 - (i * 0.001), "genre": "techno"}
                for i in range(1000)
            ]
            mock_similar.return_value = large_stem_list
            
            prompt_analysis = {
                "genre": "techno",
                "elements": ["kick", "bass"],
                "embeddings": np.random.rand(512).tolist()
            }
            
            start_time = time.time()
            selected_stems = await arranger._select_stems(
                prompt_analysis,
                session=test_db_session,
                max_stems=10
            )
            end_time = time.time()
            
            processing_time = end_time - start_time
            
            assert len(selected_stems) == 10
            assert processing_time < 2.0  # Sollte unter 2 Sekunden dauern
    
    @pytest.mark.integration
    async def test_full_arrangement_workflow(self, arranger: ArrangerService, test_db_session, sample_text_prompts):
        """Test: Vollständiger Arrangement-Workflow"""
        prompt = sample_text_prompts[0]
        
        with patch.object(arranger.neuro_analyzer, 'analyze_text_prompt') as mock_analyze, \
             patch.object(arranger.neuro_analyzer, 'get_similar_stems') as mock_similar:
            
            # Mock Setup
            mock_analyze.return_value = {
                "embeddings": np.random.rand(512).tolist(),
                "prompt": prompt
            }
            
            mock_similar.return_value = [
                {"id": i, "type": "kick", "similarity": 0.9, "genre": "techno"}
                for i in range(1, 6)
            ]
            
            # Vollständigen Workflow ausführen
            result = await arranger.create_arrangement(
                prompt=prompt,
                duration=180,
                session=test_db_session
            )
            
            # Ergebnis validieren
            assert "arrangement_id" in result
            assert "structure" in result
            assert "stems" in result
            assert "metadata" in result
            
            structure = result["structure"]
            assert "sections" in structure
            assert "total_duration" in structure
            assert structure["total_duration"] <= 180
            
            # Mindestens eine Sektion sollte vorhanden sein
            assert len(structure["sections"]) >= 1
            
            # Alle Sektionen sollten gültig sein
            for section in structure["sections"]:
                assert "name" in section
                assert "start" in section
                assert "duration" in section
                assert "stems" in section
                assert section["duration"] > 0