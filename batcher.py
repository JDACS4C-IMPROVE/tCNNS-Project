import numpy as np
from random import shuffle


class Batch():
    def __init__(self, batch_size, value, drug, cell, positions):
        self.batch_size = batch_size
        self.positions = positions
        self.value = value
        self.drug = drug
        self.cell = cell
        self.offset = 0
        self.size = positions.shape[0]

    def mini_batch(self):
        if self.offset >= self.size:
            return None
        if self.offset + self.batch_size <= self.size:
            sub_posi = self.positions[self.offset : self.offset + self.batch_size]
        else:
            sub_posi = self.positions[self.offset : ]
        self.offset += self.batch_size
        cell = []
        drug = []
        value = []
        """
        for row, col in sub_posi:
            drug.append(self.drug[row])
            cell.append(self.cell[col])
            value.append(self.value[row, col])
        """
        for didx, cidx, tidx in sub_posi:
            drug.append(self.drug[didx])
            cell.append(self.cell[cidx])
            value.append(self.value[didx,tidx])
        return np.array(value), np.array(drug), np.array(cell)
    
    def whole_batch(self):
        cell = []
        drug = []
        value = []
        for didx, cidx, tidx in self.positions:
            drug.append(self.drug[didx])
            cell.append(self.cell[cidx])
            value.append(self.value[didx,tidx])
        """
        for row, col in self.positions:
            drug.append(self.drug[row])
            cell.append(self.cell[col])
            value.append(self.value[row, col])
        """
        return np.array(value), np.array(drug), np.array(cell)
        
    def diy_batch(self, k):
        cell = []
        drug = []
        value = []
        for row, col in self.positions[range(k)]:
            drug.append(self.drug[row])
            cell.append(self.cell[col])
            value.append(self.value[row, col])
        return np.array(value), np.array(drug), np.array(cell)
    
    def reset(self):
        self.offset = 0

    def available(self):
        if self.offset < self.size:
            return True
        else:
            return False
