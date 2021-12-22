
package hopes.cic.xml;

import javax.xml.bind.annotation.XmlEnum;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for EncryptionType.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * <p>
 * <pre>
 * &lt;simpleType name="EncryptionType"&gt;
 *   &lt;restriction base="{http://www.w3.org/2001/XMLSchema}string"&gt;
 *     &lt;enumeration value="NO"/&gt;
 *     &lt;enumeration value="LEA"/&gt;
 *     &lt;enumeration value="HIGHT"/&gt;
 *     &lt;enumeration value="SEED"/&gt;
 *   &lt;/restriction&gt;
 * &lt;/simpleType&gt;
 * </pre>
 * 
 */
@XmlType(name = "EncryptionType")
@XmlEnum
public enum EncryptionType {

    NO,
    LEA,
    HIGHT,
    SEED;

    public String value() {
        return name();
    }

    public static EncryptionType fromValue(String v) {
        return valueOf(v);
    }

}
