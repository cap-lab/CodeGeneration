
package hopes.cic.xml;

import javax.xml.bind.annotation.XmlEnum;
import javax.xml.bind.annotation.XmlEnumValue;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for ChannelTypeType.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * <p>
 * <pre>
 * &lt;simpleType name="ChannelTypeType"&gt;
 *   &lt;restriction base="{http://www.w3.org/2001/XMLSchema}string"&gt;
 *     &lt;enumeration value="fifo"/&gt;
 *     &lt;enumeration value="array"/&gt;
 *     &lt;enumeration value="overwritable"/&gt;
 *   &lt;/restriction&gt;
 * &lt;/simpleType&gt;
 * </pre>
 * 
 */
@XmlType(name = "ChannelTypeType")
@XmlEnum
public enum ChannelTypeType {

    @XmlEnumValue("fifo")
    FIFO("fifo"),
    @XmlEnumValue("array")
    ARRAY("array"),
    @XmlEnumValue("overwritable")
    OVERWRITABLE("overwritable");
    private final String value;

    ChannelTypeType(String v) {
        value = v;
    }

    public String value() {
        return value;
    }

    public static ChannelTypeType fromValue(String v) {
        for (ChannelTypeType c: ChannelTypeType.values()) {
            if (c.value.equals(v)) {
                return c;
            }
        }
        throw new IllegalArgumentException(v);
    }

}
