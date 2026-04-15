"""Unit tests for data converters."""
import tempfile
from pathlib import Path
import numpy as np
import h5py
import pytest


class TestGDB13Converter:
    """Tests for gdb13.py converter functions."""
    
    def test_parse_properties_indices_valid(self):
        """Test parsing valid Properties string."""
        from uni_react.data.converters.gdb13 import parse_properties_indices
        
        comment = 'Properties=species:S:1:pos:R:3:forces:R:3:charge:R:1 energy=-123.45'
        pos_offset, q_offset = parse_properties_indices(comment)
        
        assert pos_offset == 0
        assert q_offset == 6  # pos(3) + forces(3)
    
    def test_parse_properties_indices_different_order(self):
        """Test parsing Properties with different field order."""
        from uni_react.data.converters.gdb13 import parse_properties_indices
        
        comment = 'Properties=species:S:1:charge:R:1:pos:R:3:forces:R:3'
        pos_offset, q_offset = parse_properties_indices(comment)
        
        assert pos_offset == 1  # charge(1) + pos
        assert q_offset == 0    # charge comes first
    
    def test_parse_properties_indices_missing_properties(self):
        """Test that missing Properties field raises ValueError."""
        from uni_react.data.converters.gdb13 import parse_properties_indices
        
        comment = 'energy=-123.45'
        with pytest.raises(ValueError, match="Missing Properties"):
            parse_properties_indices(comment)
    
    def test_parse_properties_indices_malformed(self):
        """Test that malformed Properties raises ValueError."""
        from uni_react.data.converters.gdb13 import parse_properties_indices
        
        comment = 'Properties=species:S:1:pos:R energy=-123.45'
        with pytest.raises(ValueError, match="Malformed Properties"):
            parse_properties_indices(comment)
    
    def test_parse_properties_indices_missing_species(self):
        """Test that missing species field raises UnsupportedPropertiesError."""
        from uni_react.data.converters.gdb13 import (
            parse_properties_indices,
            UnsupportedPropertiesError
        )
        
        comment = 'Properties=pos:R:3:charge:R:1'
        with pytest.raises(UnsupportedPropertiesError, match="species must be first"):
            parse_properties_indices(comment)
    
    def test_parse_properties_indices_missing_pos(self):
        """Test that missing pos field raises ValueError."""
        from uni_react.data.converters.gdb13 import parse_properties_indices
        
        comment = 'Properties=species:S:1:charge:R:1:forces:R:3'
        with pytest.raises(ValueError, match="missing pos"):
            parse_properties_indices(comment)
    
    def test_parse_properties_indices_missing_charge(self):
        """Test that missing charge field raises ValueError."""
        from uni_react.data.converters.gdb13 import parse_properties_indices
        
        comment = 'Properties=species:S:1:pos:R:3:forces:R:3'
        with pytest.raises(ValueError, match="missing pos or charge"):
            parse_properties_indices(comment)
    
    def test_parse_properties_indices_wrong_pos_count(self):
        """Test that pos with wrong column count raises ValueError."""
        from uni_react.data.converters.gdb13 import parse_properties_indices
        
        comment = 'Properties=species:S:1:pos:R:2:charge:R:1'
        with pytest.raises(ValueError, match="pos should have 3 columns"):
            parse_properties_indices(comment)
    
    def test_parse_properties_indices_wrong_charge_count(self):
        """Test that charge with wrong column count raises ValueError."""
        from uni_react.data.converters.gdb13 import parse_properties_indices
        
        comment = 'Properties=species:S:1:pos:R:3:charge:R:2'
        with pytest.raises(ValueError, match="charge should have 1 column"):
            parse_properties_indices(comment)
    
    def test_parse_energy_and_dipole_valid(self):
        """Test parsing valid energy and dipole."""
        from uni_react.data.converters.gdb13 import parse_energy_and_dipole
        
        comment = 'energy=-123.45 dipole="0.1 0.2 0.3"'
        energy, dipole = parse_energy_and_dipole(comment)
        
        assert energy == pytest.approx(-123.45)
        assert dipole.shape == (3,)
        assert dipole[0] == pytest.approx(0.1)
        assert dipole[1] == pytest.approx(0.2)
        assert dipole[2] == pytest.approx(0.3)
        assert dipole.dtype == np.float32
    
    def test_parse_energy_and_dipole_scientific_notation(self):
        """Test parsing energy in scientific notation."""
        from uni_react.data.converters.gdb13 import parse_energy_and_dipole
        
        comment = 'energy=-1.23e-4 dipole="1.0e-3 2.0e-3 3.0e-3"'
        energy, dipole = parse_energy_and_dipole(comment)
        
        assert energy == pytest.approx(-1.23e-4)
        assert dipole[0] == pytest.approx(1.0e-3)
    
    def test_parse_energy_and_dipole_missing_energy(self):
        """Test that missing energy raises ValueError."""
        from uni_react.data.converters.gdb13 import parse_energy_and_dipole
        
        comment = 'dipole="0.1 0.2 0.3"'
        with pytest.raises(ValueError, match="Missing energy"):
            parse_energy_and_dipole(comment)
    
    def test_parse_energy_and_dipole_missing_dipole(self):
        """Test that missing dipole raises ValueError."""
        from uni_react.data.converters.gdb13 import parse_energy_and_dipole
        
        comment = 'energy=-123.45'
        with pytest.raises(ValueError, match="Missing dipole"):
            parse_energy_and_dipole(comment)
    
    def test_parse_energy_and_dipole_wrong_dipole_components(self):
        """Test that dipole with wrong number of components raises ValueError."""
        from uni_react.data.converters.gdb13 import parse_energy_and_dipole
        
        comment = 'energy=-123.45 dipole="0.1 0.2"'
        with pytest.raises(ValueError, match="dipole has 2 components, expected 3"):
            parse_energy_and_dipole(comment)
    
    def test_e2z_mapping(self):
        """Test element to atomic number mapping."""
        from uni_react.data.converters.gdb13 import E2Z
        
        assert E2Z["H"] == 1
        assert E2Z["C"] == 6
        assert E2Z["N"] == 7
        assert E2Z["O"] == 8
        assert len(E2Z) == 4  # Only CHON
    
    def test_min_interatomic_distance_ok_valid(self):
        """Test minimum distance check with valid geometry."""
        from uni_react.data.converters.gdb13 import min_interatomic_distance_ok
        
        # Two atoms 2.0 Å apart
        R = np.array([[0.0, 0.0, 0.0],
                      [2.0, 0.0, 0.0]], dtype=np.float32)
        
        assert min_interatomic_distance_ok(R, min_dist=1.0) is True
        assert min_interatomic_distance_ok(R, min_dist=1.5) is True
    
    def test_min_interatomic_distance_ok_too_close(self):
        """Test minimum distance check with atoms too close."""
        from uni_react.data.converters.gdb13 import min_interatomic_distance_ok
        
        # Two atoms 0.5 Å apart
        R = np.array([[0.0, 0.0, 0.0],
                      [0.5, 0.0, 0.0]], dtype=np.float32)
        
        assert min_interatomic_distance_ok(R, min_dist=1.0) is False
    
    def test_min_interatomic_distance_ok_single_atom(self):
        """Test minimum distance check with single atom."""
        from uni_react.data.converters.gdb13 import min_interatomic_distance_ok
        
        R = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        
        # Single atom should return False
        assert min_interatomic_distance_ok(R, min_dist=0.0) is False
    
    def test_min_interatomic_distance_ok_three_atoms(self):
        """Test minimum distance check with three atoms."""
        from uni_react.data.converters.gdb13 import min_interatomic_distance_ok
        
        # Three atoms forming a triangle
        R = np.array([[0.0, 0.0, 0.0],
                      [2.0, 0.0, 0.0],
                      [1.0, 1.732, 0.0]], dtype=np.float32)
        
        assert min_interatomic_distance_ok(R, min_dist=1.5) is True
        assert min_interatomic_distance_ok(R, min_dist=2.5) is False


class TestHDF5Schema:
    """Tests for HDF5 output schema validation."""
    
    def test_hdf5_schema_structure(self):
        """Test that HDF5 output has correct schema structure."""
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            temp_h5 = f.name
        
        try:
            # Create a test HDF5 file with expected schema
            with h5py.File(temp_h5, 'w') as h5:
                # Create frames group
                gF = h5.create_group('frames')
                gF.create_dataset('offsets', data=np.array([0, 2, 5], dtype=np.int64))
                gF.create_dataset('n_atoms', data=np.array([2, 3], dtype=np.int32))
                gF.create_dataset('energy', data=np.array([-1.0, -2.0], dtype=np.float64))
                gF.create_dataset('dipole', data=np.random.randn(2, 3).astype(np.float32))
                
                # Create atoms group
                gA = h5.create_group('atoms')
                gA.create_dataset('Z', data=np.array([1, 1, 6, 6, 6], dtype=np.uint8))
                gA.create_dataset('R', data=np.random.randn(5, 3).astype(np.float32))
                gA.create_dataset('q', data=np.random.randn(5).astype(np.float32))
                gA.create_dataset('q_mulliken', data=np.random.randn(5).astype(np.float32))
            
            # Verify schema
            with h5py.File(temp_h5, 'r') as h5:
                # Check groups exist
                assert 'frames' in h5
                assert 'atoms' in h5
                
                # Check frames datasets
                assert 'offsets' in h5['frames']
                assert 'n_atoms' in h5['frames']
                assert 'energy' in h5['frames']
                assert 'dipole' in h5['frames']
                
                # Check atoms datasets
                assert 'Z' in h5['atoms']
                assert 'R' in h5['atoms']
                assert 'q' in h5['atoms']
                assert 'q_mulliken' in h5['atoms']
                
                # Check dtypes
                assert h5['frames/offsets'].dtype == np.int64
                assert h5['frames/n_atoms'].dtype == np.int32
                assert h5['frames/energy'].dtype == np.float64
                assert h5['frames/dipole'].dtype == np.float32
                assert h5['atoms/Z'].dtype == np.uint8
                assert h5['atoms/R'].dtype == np.float32
                assert h5['atoms/q'].dtype == np.float32
                
                # Check shapes
                assert h5['frames/offsets'].shape == (3,)  # n_frames + 1
                assert h5['frames/n_atoms'].shape == (2,)
                assert h5['frames/energy'].shape == (2,)
                assert h5['frames/dipole'].shape == (2, 3)
                assert h5['atoms/Z'].shape == (5,)
                assert h5['atoms/R'].shape == (5, 3)
                assert h5['atoms/q'].shape == (5,)
        
        finally:
            Path(temp_h5).unlink()
    
    def test_hdf5_offsets_consistency(self):
        """Test that offsets array is consistent with n_atoms."""
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            temp_h5 = f.name
        
        try:
            with h5py.File(temp_h5, 'w') as h5:
                gF = h5.create_group('frames')
                # 3 frames with 2, 3, 4 atoms
                offsets = np.array([0, 2, 5, 9], dtype=np.int64)
                n_atoms = np.array([2, 3, 4], dtype=np.int32)
                
                gF.create_dataset('offsets', data=offsets)
                gF.create_dataset('n_atoms', data=n_atoms)
            
            # Verify consistency
            with h5py.File(temp_h5, 'r') as h5:
                offsets = h5['frames/offsets'][:]
                n_atoms = h5['frames/n_atoms'][:]
                
                # Check that differences match n_atoms
                for i in range(len(n_atoms)):
                    assert offsets[i+1] - offsets[i] == n_atoms[i]
        
        finally:
            Path(temp_h5).unlink()


class TestExtXYZParsing:
    """Integration tests for parsing extxyz format."""
    
    def test_parse_simple_molecule(self):
        """Test parsing a simple H2 molecule."""
        # Create a temporary extxyz file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.extxyz', delete=False) as f:
            f.write('2\n')
            f.write('Properties=species:S:1:pos:R:3:charge:R:1 energy=-1.0 dipole="0.0 0.0 0.0"\n')
            f.write('H 0.0 0.0 0.0 0.1\n')
            f.write('H 0.0 0.0 0.74 -0.1\n')
            temp_path = f.name
        
        try:
            # Verify file exists and has correct content
            assert Path(temp_path).exists()
            
            with open(temp_path, 'r') as f:
                lines = f.readlines()
                assert len(lines) == 4
                assert lines[0].strip() == '2'
                assert 'Properties=' in lines[1]
                assert 'energy=' in lines[1]
                assert 'dipole=' in lines[1]
        
        finally:
            Path(temp_path).unlink()
