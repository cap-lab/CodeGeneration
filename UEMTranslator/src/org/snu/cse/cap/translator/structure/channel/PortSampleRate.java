package org.snu.cse.cap.translator.structure.channel;

public class PortSampleRate {
	private String modeName;
	private int sampleRate;
	private int maxAvailableNum;
	
	public String getModeName() {
		return modeName;
	}
	
	public void setModeName(String modeName) {
		this.modeName = modeName;
	}
	
	public int getSampleRate() {
		return sampleRate;
	}
	
	public void setSampleRate(int sampleRate) {
		this.sampleRate = sampleRate;
	}
	
	public int getMaxAvailableNum() {
		return maxAvailableNum;
	}
	
	public void setMaxAvailableNum(int maxAvailableNum) {
		this.maxAvailableNum = maxAvailableNum;
	}
}
