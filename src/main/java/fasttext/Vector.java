package fasttext;

import com.google.common.base.Preconditions;

public class Vector {

  final int m;
  final float[] data;

  public Vector(final int size) {
    this.m = size;
    this.data = new float[size];
  }

  public int size() {
    return this.m;
  }

  public void zero() {
    for (int i = 0; i < m; i++) {
      data[i] = 0.0f;
    }
  }

  public float norm() {
    float sum = 0.0f;
    for (int i = 0; i < m; i++) {
      sum += data[i] * data[i];
    }
    return (float) Math.sqrt(sum);
  }

  public void mul(final float a) {
    for (int i = 0; i < m; i++) {
      data[i] *= a;
    }
  }

  public float dot(final Vector vector) {
    float dist = 0;
    for (int i = 0; i < data.length; i++) {
      dist += vector.at(i) * data[i];
    }
    return dist;
  }

  public void addVector(final Vector source) {
    Preconditions.checkArgument(source.size() == m);
    for (int i = 0; i < m; i++) {
      data[i] += source.at(i);
    }
  }

  public void addVector(final Vector source, final float s) {
    Preconditions.checkArgument(source.size() == m);
    for (int i = 0; i < m; i++) {
      data[i] += s * source.at(i);
    }
  }

  public void addRow(final Matrix matrix, final int i, final float a) {
    Preconditions.checkPositionIndex(i, matrix.m());
    Preconditions.checkArgument(m == matrix.n());
    for (int j = 0; j < matrix.n(); j++) {
      data[j] += a * matrix.at(i, j);
    }
  }

  public void addRow(final Matrix matrix, final int i) {
    Preconditions.checkPositionIndex(i, matrix.m());
    Preconditions.checkArgument(m == matrix.n());
    for (int j = 0; j < matrix.n(); j++) {
      data[j] += matrix.at(i, j);
    }
  }

  public void mul(final Matrix matrix, final Vector vector) {
    Preconditions.checkArgument(m == matrix.m());
    Preconditions.checkArgument(matrix.n() == vector.size());
    for (int i = 0; i < m; i++) {
      data[i] = matrix.dotRow(vector, i);
    }
  }

  public int argmax() {
    float max = data[0];
    int argmax = 0;
    for (int i = 1; i < m; i++) {
      if (data[i] > max) {
        max = data[i];
        argmax = i;
      }
    }
    return argmax;
  }

  public void addAt(final int i, final float v) {
    Preconditions.checkPositionIndex(i, m);
    data[i] += v;
  }

  public void set(final int i, final float v) {
    Preconditions.checkPositionIndex(i, m);
    data[i] = v;
  }

  public float at(final int i) {
    Preconditions.checkPositionIndex(i, m);
    return data[i];
  }

  @Override
  public String toString() {
    final StringBuilder builder = new StringBuilder();
    builder.append("Vector(");
    builder.append("m=");
    builder.append(m);
    builder.append(", data=");
    builder.append("[");
    for (float d : data) {
      builder.append(d).append(' ');
    }
    if (builder.length() > 1) {
      builder.setLength(builder.length() - 1);
    }
    builder.append("]");

    builder.append(")");
    return builder.toString();
  }

  public float[] toArray() {
    return this.data;
  }

}
