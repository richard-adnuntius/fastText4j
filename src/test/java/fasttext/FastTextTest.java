package fasttext;

import junit.framework.TestCase;

public class FastTextTest extends TestCase {

    public void testGetWordVector() throws Exception {
        FastTextModel model = FastText.loadModel("/Users/ramast/development/fasttext-models/en/cc.en.300.bin");

        final Vector hello = model.getWordVector("hello");
        final Vector bye = model.getWordVector("bye");
        System.out.println(compare(hello, bye));

        System.out.println(compare(model.getWordVector("red"), model.getWordVector("blue")));
        System.out.println(compare(model.getWordVector("cheese"), model.getWordVector("cheddar")));
        System.out.println(compare(model.getWordVector("imagination"), model.getWordVector("tall")));
    }

    public static float compare(final Vector v1, final Vector v2) {
        v1.mul(1.0f / v1.norm());
        v2.mul(1.0f / v2.norm());

        float dist = 0;
        for (int i = 0; i < v1.size(); i++) {
            dist += v1.at(i) * v2.at(i);
        }
        return dist;
    }

}