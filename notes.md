## Ideas

Consider that instead of using min as a prefix, we could also use GCD, which might be more
efficient for creating smaller associated values in some cases.

Also consider creating a fixed size FST, it will increase the speed of insertion a lot at the
cost of compile time requirements.
