package fasttext;

import com.google.common.primitives.UnsignedLong;
import fasttext.store.InputStreamFastTextInput;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class FastTextModel {

    static final String EOS = "</s>";
    static final String BEGINNING_OF_WORD = "<";
    static final String END_OF_WORD = ">";

    private final Args args;
    private final Map<String, Entry> words;
    private final Matrix wordData;
    private final Matrix ngramData;

    protected FastTextModel(final Args args,
                            final Map<String, Entry> words,
                            final Matrix wordData,
                            final Matrix ngramData) {
        this.args = args;
        this.ngramData = ngramData;
        this.words = words;
        this.wordData = wordData;
    }

    /**
     * Gets the vector for a word.
     */
    public Vector getWordVector(String word) {
        final Vector vector = new Vector(args.getDimension());
        vector.zero();
        final FastTextModel.Entry entry = words.get(word);
        List<Integer> ngrams = null;
        if (entry != null) {
            vector.addRow(wordData, entry.index);
            if (entry.ngrams != null) {
                ngrams = entry.ngrams;
            }
        }
        if (ngrams == null) {
            ngrams = getNgrams(word);
        }
        for (final int it : ngrams) {
            vector.addRow(ngramData, it);
        }
        if (!ngrams.isEmpty()) {
            vector.mul(1.0f / ((float) ngrams.size() + 1));
        }
        return vector;
    }

    public List<Integer> getNgrams(String word) {
        final Entry entry = words.get(word);
        if (entry == null || entry.ngrams == null) {
            final List<Integer> ngrams = new ArrayList<>();
            if (!word.equals(EOS)) {
                computeNgrams(BEGINNING_OF_WORD + word + END_OF_WORD, ngrams);
            }
            if (entry == null) {
                // The word is not in the dictionary.
                return ngrams;
            } else {
                entry.ngrams = ngrams;
            }
        }
        return entry.ngrams;
    }

    /**
     * String FNV-1a 32 bits Hash
     */
    public long hash(final String str) {
        int h = (int) 2166136261L;      // 0xffffffc5;
        for (byte strByte : str.getBytes()) {
            h = (h ^ strByte) * 16777619; // FNV-1a
        }
        return h & 0xffffffffL;
    }

    protected boolean hasUTFContinuationCode(char c) {
        return (c & 0xC0) == 0x80;
    }

    protected void computeNgrams(String word, List<Integer> ngrams) {
        for (int i = 0; i < word.length(); i++) {
            final StringBuilder ngram = new StringBuilder();
            if (!hasUTFContinuationCode(word.charAt(i))) {
                for (int j = i, n = 1; j < word.length() && n <= args.getMaxn(); n++) {
                    do {
                        ngram.append(word.charAt(j++));
                    } while (j < word.length() && hasUTFContinuationCode(word.charAt(j)));
                    if (n >= args.getMinn() && !(n == 1 && (i == 0 || j == word.length()))) {
                        final UnsignedLong h = UnsignedLong.valueOf(hash(ngram.toString()));
                        ngrams.add(h.mod(UnsignedLong.valueOf(args.getBucketNumber())).intValue());
                    }
                }
            }
        }
    }

    public static FastTextModel load(Args args, InputStreamFastTextInput is) throws IOException {
        final int size = is.readInt();
        is.readInt();
        is.readInt();
        is.readLong();
        is.readLong();

        final Map<String, Entry> words = new HashMap<>(size);
        for (int i = 0; i < size; i++) {
            final String word = is.readString();
            // These are not used - just skip over them
            is.readLong();
            is.readByteAsInt();
            words.put(word, new Entry(i));
        }

        final boolean quant = is.readBoolean();
        if (quant) {
            throw new IllegalArgumentException("Cannot load quantised models");
        }

        final int m = (int) is.readLong();
        final int n = (int) is.readLong();
        final float[] dictionaryData = new float[m * n / 2];
        for (int i = 0; i < m * n / 2; i++) {
            dictionaryData[i] = is.readFloat();
        }
        final Matrix wordVectors = new Matrix(m / 2, n, dictionaryData);

        System.out.println("Loaded dictionary vectors");

        final float[] ngramData = new float[m * n / 2];
        for (int i = 0; i < m * n / 2; i++) {
            ngramData[i] = is.readFloat();
        }
        System.out.println("Loaded ngram vectors");

        final Matrix ngramVectors = new Matrix(m / 2, n, ngramData);

        // Skip unused field
        is.readBoolean();

        return new FastTextModel(args, words, wordVectors, ngramVectors);
    }

    public static class Entry {
        /**
         * The row index into the word vector matrix.
         */
        final int index;

        /**
         * This is lazily loaded when an entry is accessed.
         * They are row indices into the ngram matrix.
         */
        List<Integer> ngrams = null;

        public Entry(final int index) {
            this.index = index;
        }
    }

}