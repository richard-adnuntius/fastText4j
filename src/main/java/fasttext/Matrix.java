package fasttext;

import com.google.common.base.Preconditions;

public class Matrix {

    private final float[] data;
    private final int m;
    private final int n;

    public Matrix(int m, int n, float[] data) {
        this.m = m;
        this.n = n;
        this.data = data;
    }

    public Matrix(int m, int n) {
        this.m = m;
        this.n = n;
        data = new float[this.m * this.n];
    }

    public Matrix(Matrix other) {
        this.m = other.m;
        this.n = other.n;
        data = new float[this.m * this.n];
        for (int i = 0; i < (this.m * this.n); i++) {
            data[i] = other.data[i];
        }
    }

    public void zero() {
        for (int i = 0; i < (m * n); i++) {
            data[i] = 0.0f;
        }
    }

    public float[] atRow(int i) {
        float[] r = new float[n];
        System.arraycopy(data, i * n, r, 0, n);
        return r;
    }

    public float at(int i, int j) {
        return data[i * n + j];
    }

    public float dotRow(final Vector vec, int i) {
        Preconditions.checkPositionIndex(i, m);
        Preconditions.checkArgument(vec.size() == n);
        float d = 0.0f;
        for (int j = 0; j < n; j++) {
            d += data[i * n + j] * vec.at(j);
        }
        if (Float.isNaN(d)) {
            throw new IllegalStateException("Encountered NaN.");
        }
        return d;
    }

    public void addRow(final Vector vec, int i, float a) {
        Preconditions.checkPositionIndex(i, m);
        Preconditions.checkArgument(vec.size() == n);
        for (int j = 0; j < n; j++) {
            data[i * n + j] += a * vec.at(j);
        }
    }

    public void multiplyRow(final Vector nums) {
        multiplyRow(nums, 0, -1);
    }

    public void multiplyRow(final Vector nums, int ib, int ie) {
        if (ie == -1) {
            ie = m;
        }
        Preconditions.checkPositionIndex(ie, nums.size());
        for (int i = ib; i < ie; i++) {
            float num = nums.at(i - ib);
            if (n != 0) {
                for (int j = 0; j < this.n; j++) {
                    data[i * n + j] *= num;
                }
            }
        }
    }

    public void divideRow(final Vector denoms) {
        divideRow(denoms, 0, -1);
    }

    public void divideRow(final Vector denoms, int ib, int ie) {
        if (ie == -1) {
            ie = m;
        }
        Preconditions.checkPositionIndex(ie, denoms.size());
        for (int i = ib; i < ie; i++) {
            float denom = denoms.at(i - ib);
            if (denom != 0) {
                for (int j = 0; j < this.n; j++) {
                    data[i * n + j] /= denom;
                }
            }
        }
    }

    public int m() {
        return this.m;
    }

    public int n() {
        return this.n;
    }
}
