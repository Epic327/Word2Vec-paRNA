import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--seqsPath',type=str)
parser.add_argument('--k',type=int)
parser.add_argument('--s',type=int)
parser.add_argument('--destPath',type=str)

args = parser.parse_args()

seqs = args.seqsPath
k = args.k
s = args.s
dest = args.destPath


def seq2ngram2(seqs, k, s, dest):   #如果num《100000   ，dest:所有序列的k-mer 返回的是pos对应的mer，或者neg对应的mer
    f = open(seqs)
    lines = f.readlines()
    f.close()

    print('need to n-gram %d lines' % len(lines))
    f = open(dest, 'w')
    for num, line in enumerate(lines):
        if num < 100000:
            line = line[:-1].lower()  # remove '\n' and lower ACGT
            l = len(line)  # length of line

            for i in range(0, l, s):
                if i + k >= l + 1:
                    break

                f.write(''.join(line[i:i + k]))
                f.write(' ')
            f.write('\n')
    f.close()

if __name__ == '__main__':
    seq2ngram2('./text-data/'+seqs, k, s, './kmer-data/'+dest)