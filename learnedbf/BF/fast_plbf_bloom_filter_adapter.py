from learnedbf.BF import ClassicalBloomFilter

class BloomFilter:
    def __init__(self, max_elements: int, error_rate: float):
        self.bloom_filter = ClassicalBloomFilter(epsilon=error_rate, 
                                                 n=max_elements)
        
    def add(self, item):
        self.bloom_filter.add(item)
    
    def __contains__(self, item):
        return self.bloom_filter.check(item)

    def get_size(self):
        return self.bloom_filter.get_size()
