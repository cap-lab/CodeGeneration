package org.snu.cse.cap.translator.structure.device;

public class NoProcessorFoundException extends Exception {

	/**
	 * 
	 */
	private static final long serialVersionUID = 5722876674594804569L;
	
	public NoProcessorFoundException() {
		
	}
	
	public NoProcessorFoundException(String message) {
		super(message);
	}
}
