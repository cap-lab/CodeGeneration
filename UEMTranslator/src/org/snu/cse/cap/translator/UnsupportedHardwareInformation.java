package org.snu.cse.cap.translator;

public class UnsupportedHardwareInformation extends Exception {

	/**
	 * 
	 */
	private static final long serialVersionUID = 7757248592966214691L;
	
	public UnsupportedHardwareInformation() {
		
	}
	
	public UnsupportedHardwareInformation(String message) {
		super(message);
	}
}
