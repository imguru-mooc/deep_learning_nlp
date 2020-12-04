'''
2019-08-07
Saltlux AI Labs @ AIR group
Seonghyun Kim
원본 출처 : https://github.com/lovit/WordPieceModel 
한국어를 대상으로 한 wordPiece 코드를 수정
'''


from collections import Counter
from collections import defaultdict
import argparse

def load_corpus(fname):
    data=[]
    with open(fname, 'r', encoding="utf-8") as raw:
        for r in raw:
            if r !='\n':
                data.append(r[:-1])
        return data

class BytePairEncoder:
    
    def __init__(self, n_iters=10, verbose=True):
        self.n_iters = n_iters if n_iters > 0 else 10
        self.units = {}
        self.max_length = 0
        self.verbose = verbose
        
    def train(self, sents):
        if self.verbose:
            print('begin vocabulary scanning', end='', flush=True)
        
        vocabs = self._sent_to_vocabs(sents)
        if self.verbose:
            print('\rterminated vocabulary scanning', flush=True)
        
        self.units = self._build_subword_units(vocabs)
    
    def _sent_to_vocabs(self, sents):        
        vocabs = Counter((eojeol.replace('_', '') for sent in sents for eojeol in sent.split() if eojeol))
        return {' ##'.join(w) : c for w,c in vocabs.items() if w}
        
    def _build_subword_units(self, vocabs):
        def get_stats(vocabs):
            pairs = defaultdict(int)
            for word, freq in vocabs.items():
                symbols = word.split()
                for i in range(len(symbols)-1):
                    pairs[(symbols[i],symbols[i+1])] += freq
            return pairs
        
        def merge_vocab(pair, v_in):
            v_out = {}
            bigram = ' '.join(pair)
            back_pair = pair[1]
            replacer = pair[0] +  back_pair.replace("##","")
            for word, freq in v_in.items():
                w_out = word.replace(bigram, replacer)
                v_out[w_out] = freq
            return v_out
        
        for i in range(self.n_iters + 1):
            pairs = get_stats(vocabs)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            print(best)
            vocabs = merge_vocab(best, vocabs)
            if self.verbose and i % 100 == 99:
                print('\rtraining bpe {} / {}'.format(i+1, self.n_iters), end='', flush=True)
        if self.verbose:
            print('\rtraining bpe was done{}'.format(' '*40))
        
        units = {}
        for word, freq in vocabs.items():
            for unit in word.split():
                units[unit] = units.get(unit, 0) + freq
        self.max_length = max((len(w) for w in units))
        return units
    
    def save(self, fname):
        with open(fname, 'w', encoding='utf-8') as f:
            f.write('[CLS]\n[UNK]\n[SEP]\n[PAD]\n[MASK]\n')
            for unit, frequency in sorted(self.units.items(), key=lambda x:(-x[1], -len(x[0]))):
                f.write('{}\n'.format(unit))
            #f.write()
                
    def load(self, fname):
        with open(fname, encoding='utf-8') as f:
            try:
                self.n_iters = int(next(f).strip().split('=')[1])
                self.max_length = int(next(f).strip().split('=')[1])
            except Exception as e:
                print(e)
            
            self.units = {}
            for row in f:
                try:
                    unit, frequency = row.strip().split('\t')
                    self.units[unit] = int(frequency)
                except Exception as e:
                    print('BPE load exception: {}'.format(str(e)))
                    break
                    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='wordpiece model')

    parser.add_argument('--corpus', type=str)
    parser.add_argument('--iter', type=int, default=10000)
    parser.add_argument('--fname', type=str, default="vocab_tokenized_mini_10000.txt")

    args = parser.parse_args()
    encoder = BytePairEncoder(args.iter)
    encoder.train(load_corpus(args.corpus))
    encoder.save(args.fname)
