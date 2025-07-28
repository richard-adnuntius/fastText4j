package fasttext;

import fasttext.store.InputStreamFastTextInput;

import java.io.IOException;

public class Args {

  private final int dim;
  private final ModelName model;
  private final int bucket;
  private final int minN;
  private int maxN;

  private Args(int dim, ModelName model, int bucket, int minN, int maxN) {
    this.dim = dim;
    this.model = model;
    this.bucket = bucket;
    this.minN = minN;
    this.maxN = maxN;
  }

  public int getDimension() {
    return this.dim;
  }

  public ModelName getModel() {
    return this.model;
  }

  public int getBucketNumber() {
    return this.bucket;
  }

  public int getMinN() {
    return this.minN;
  }

  public int getMaxN() {
    return this.maxN;
  }

  public void setMaxN(int maxN) {
    this.maxN = maxN;
  }

  public static Args load(InputStreamFastTextInput is) throws IOException {
    int dim = is.readInt();
    is.readInt();
    is.readInt();
    is.readInt();
    is.readInt();
    is.readInt();
    is.readInt();
    ModelName model = ModelName.fromValue(is.readInt());
    int bucket = is.readInt();
    int minN = is.readInt();
    int maxN = is.readInt();
    is.readInt();
    is.readDouble();
    return new Args(dim, model, bucket, minN, maxN);
  }

  public enum ModelName {
    CBOW(1),
    SG(2),
    SUP(3);

    private final int value;

    ModelName(int value) {
      this.value = value;
    }

    public int getValue() {
      return this.value;
    }

    public static ModelName fromValue(int value) throws IllegalArgumentException {
      try {
        value -= 1;
        return ModelName.values()[value];
      } catch (ArrayIndexOutOfBoundsException e) {
        throw new IllegalArgumentException("Unknown model_name enum value :" + value);
      }
    }

  }
}