package bufferOpt.internal;

public class OptTable {
	private String mapping;
	private int OptSize;
	public OptTable(String in, int size) {
		this.mapping = in;
		this.OptSize = size;
	}
	public String getString() { return this.mapping; }
	public int getSize() { return this.OptSize; }	
}
