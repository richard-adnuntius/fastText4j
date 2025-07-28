package fasttext;

import fasttext.store.InputStreamFastTextInput;
import org.apache.log4j.Logger;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;

/**
 * Java FastText implementation.
 */
public class FastText {

    /**
     * Version 1b.
     */
    private static final int VERSION = 12;
    private static final int FILE_FORMAT_MAGIC_INT = 793712314;

    private static final Logger LOG = Logger.getLogger(FastText.class);

    private static boolean checkModel(int magic, int version) {
        if (magic != FILE_FORMAT_MAGIC_INT) {
            LOG.error("Unhandled file format");
            return false;
        }
        if (version > VERSION) {
            LOG.error("Input model version (" + version + ") doesn't match current version (" + VERSION + ")");
            return false;
        }
        return true;
    }

    /**
     * Load fastText model from file path.
     */
    public static FastTextModel loadModel(String filename) throws IOException {
        final File f = new File(filename);
        LOG.info("Loading in-memory FastText model from: " + filename);
        if (!f.canRead()) {
            throw new IllegalArgumentException("Model file cannot be opened for loading");
        }
        try (final InputStream is = Files.newInputStream(f.toPath())) {
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
            LOG.info("Loading model arguments");
            final Args args = Args.load(is);
            if (version == 11) {
                // backward compatibility: old supervised models do not use char ngrams.
                if (args.getModel() == Args.ModelName.SUP) {
                    args.setMaxN(0);
                }
            }
            LOG.info("Loading model");
            final FastTextModel model = FastTextModel.load(args, is);

            LOG.info("FastText model loaded");
            return model;
        }
    }
}