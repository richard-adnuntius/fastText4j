package fasttext;

import fasttext.store.InputStreamFastTextInput;
import org.apache.log4j.Logger;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.util.Locale;

/**
 * Java FastText implementation.
 */
public class FastText {

    public static int FASTTEXT_VERSION = 12; /* Version 1b */
    public static int FASTTEXT_FILEFORMAT_MAGIC_INT = 793712314;

    private final static Logger logger = Logger.getLogger(FastText.class.getName());

    private static boolean checkModel(int magic, int version) {
        if (magic != FASTTEXT_FILEFORMAT_MAGIC_INT) {
            logger.error("Unhandled file format");
            return false;
        }
        if (version > FASTTEXT_VERSION) {
            logger.error("Input model version (" + version + ") doesn't match current version (" + FASTTEXT_VERSION + ")");
            return false;
        }
        return true;
    }

    /**
     * Load fastText model from file path.
     * If the file is a directory, it tries to load a memory-mapped model.
     * If it is a single file, it tries to open an in-memory fastText model from binary model.
     */
    public static FastTextModel loadModel(String filename) throws IOException {
        final File f = new File(filename);
        System.out.println("Loading in-memory FastText model from:" + filename);
        if (!f.canRead()) {
            throw new IllegalArgumentException("Model file cannot be opened for loading");
        }
        try (InputStream is = Files.newInputStream(f.toPath())) {
            return loadModel(is);
        }
    }

    /**
     * Load a fastText model from a fastText binary format, reading from InputStream in.
     */
    public static FastTextModel loadModel(InputStream in) throws IOException {
        try (final InputStreamFastTextInput is = new InputStreamFastTextInput(in)) {
            final int magic = is.readInt();
            final int version = is.readInt();
            if (!checkModel(magic, version)) {
                throw new IllegalArgumentException("Model file has wrong file format");
            }
            long start = System.nanoTime();
            System.out.println("Loading model arguments");
            Args args = Args.load(is);
            if (version == 11) {
                // backward compatibility: old supervised models do not use char ngrams.
                if (args.getModel() == Args.ModelName.SUP) {
                    args.setMaxn(0);
                }
                // backward compatibility: use max vocabulary size as word2intSize.
                args.setUseMaxVocabularySize(true);
            }
            System.out.println("Loading dictionary");
            final FastTextModel model = FastTextModel.load(args, is);

            long end = System.nanoTime();
            double took = (end - start) / 1000000000d;
            System.out.println(String.format(Locale.ENGLISH, "FastText model loaded (%.3fs)", took));
            return model;
        }
    }
}