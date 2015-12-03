package hopes.cic.exception;

public class CICXMLException extends Exception {
	private static final long serialVersionUID = 1012231456101725103L;

	CICXMLErrorCode errorCode;
	
	public CICXMLException(CICXMLErrorCode errorCode) {
		this.errorCode = errorCode;
	}
	
	public CICXMLException(CICXMLErrorCode errorCode, String msg) {
		super(msg);
		this.errorCode = errorCode;
	}

	public CICXMLException(CICXMLErrorCode errorCode, Exception cause) {
		super(cause);
		this.errorCode = errorCode;
	}

	public CICXMLException(CICXMLErrorCode errorCode, String msg, Exception cause) {
		super(msg, cause);
		this.errorCode = errorCode;
	}

	public CICXMLErrorCode getErrorCode() {
		return errorCode;
	}

	@Override
	public String toString() {
		StringBuilder builder = new StringBuilder();
		
		builder.append("ErrorCode : ").append(errorCode).append("\t");
		builder.append(super.toString());
		
		return builder.toString();
	}

	
}
